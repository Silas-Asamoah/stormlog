// Opt-in CUPTI Activity injection for bounded Stormlog trace sidecars.

#include <cupti.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>
#include <zlib.h>

#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr char kHelperVersion[] = "0.4.4";
constexpr char kTraceEncoding[] = "stormlog-zlib-frames-v1";
constexpr char kTraceMagic[] = "SLCPTZ1\n";
constexpr std::size_t kFrameBufferBytes = 64U * 1024U;
constexpr std::uint32_t kCompressionLevel = 3;
constexpr std::uint64_t kFrameHeaderBytes = 12;
constexpr char kOutputDirectoryEnv[] = "STORMLOG_CUPTI_OUTPUT_DIR";
constexpr char kMaximumBytesEnv[] = "STORMLOG_CUPTI_MAX_BYTES";
constexpr char kActivitiesEnv[] = "STORMLOG_CUPTI_ACTIVITIES";
constexpr char kBufferBytesEnv[] = "STORMLOG_CUPTI_BUFFER_BYTES";
constexpr char kMaximumBufferBytesEnv[] = "STORMLOG_CUPTI_MAX_BUFFER_BYTES";
constexpr char kConsumerDelayEnv[] = "STORMLOG_CUPTI_CONSUMER_DELAY_MS";
constexpr char kDeferStartEnv[] = "STORMLOG_CUPTI_DEFER_START";
constexpr char kDeviceBufferBytesEnv[] = "STORMLOG_CUPTI_DEVICE_BUFFER_BYTES";
constexpr char kDeviceBufferPoolLimitEnv[] =
    "STORMLOG_CUPTI_DEVICE_BUFFER_POOL_LIMIT";
constexpr char kFlushPeriodEnv[] = "STORMLOG_CUPTI_FLUSH_PERIOD_MS";
constexpr char kDiscardEnv[] = "STORMLOG_CUPTI_DISCARD_RECORDS";
constexpr char kControlFilename[] = "stop.sock";
constexpr char kTracePartialFilename[] = "activity.sclz.partial";
constexpr char kTraceFilename[] = "activity.sclz";
constexpr char kStatusFilename[] = "cupti_status.json";
constexpr char kStatusTemporaryFilename[] = "cupti_status.json.tmp";
constexpr std::size_t kDefaultActivityBufferBytes = 8U * 1024U * 1024U;
constexpr std::uint64_t kDefaultMaximumBufferBytes = 32U * 1024U * 1024U;

struct ActivitySelection {
  const char* name;
  CUpti_ActivityKind kind;
};

constexpr ActivitySelection kSupportedActivities[] = {
    {"driver", CUPTI_ACTIVITY_KIND_DRIVER},
    {"runtime", CUPTI_ACTIVITY_KIND_RUNTIME},
    {"kernel", CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL},
    {"memcpy", CUPTI_ACTIVITY_KIND_MEMCPY},
    {"memset", CUPTI_ACTIVITY_KIND_MEMSET},
    {"synchronization", CUPTI_ACTIVITY_KIND_SYNCHRONIZATION},
};

struct CaptureState {
  std::mutex output_mutex;
  std::mutex error_mutex;
  std::atomic<std::uint64_t> next_record_id{1};
  std::atomic<std::uint64_t> delivered_records{0};
  std::atomic<std::uint64_t> cupti_dropped_records{0};
  std::atomic<std::uint64_t> local_dropped_records{0};
  std::atomic<std::uint64_t> bytes_written{0};
  std::atomic<std::uint64_t> bytes_dropped{0};
  std::atomic<std::uint64_t> uncompressed_bytes_delivered{0};
  std::atomic<std::uint64_t> buffers_requested{0};
  std::atomic<std::uint64_t> outstanding_buffer_bytes{0};
  std::atomic<std::uint64_t> peak_outstanding_buffer_bytes{0};
  std::atomic<std::uint64_t> buffer_completed_ns{0};
  std::atomic<std::uint64_t> kernels_without_timestamps{0};
  // Activity buffers CUPTI asked for while the outstanding cap was reached;
  // CUPTI drops the records it would have written to them.
  std::atomic<std::uint64_t> buffer_refusals{0};
  std::uint64_t maximum_buffer_bytes{kDefaultMaximumBufferBytes};
  std::uint64_t device_buffer_bytes{0};
  std::uint64_t device_buffer_pool_limit{0};
  std::uint64_t flush_period_ms{0};
  // Ablation only: count records without serializing them.
  bool discard_records{false};
  std::atomic<std::uint64_t> discarded_records{0};
  int directory_fd{-1};
  int trace_fd{-1};
  std::atomic<int> control_fd{-1};
  std::thread control_thread;
  // Completed buffers wait here for the writer thread, so CUPTI's callback
  // never decodes, encodes or compresses records.
  std::mutex queue_mutex;
  std::condition_variable queue_ready;
  std::deque<std::pair<std::uint8_t*, std::size_t>> completed_buffers;
  bool writer_stopping{false};
  std::thread writer_thread;
  CUpti_SubscriberHandle subscriber{nullptr};
  std::string output_directory;
  std::string control_path;
  std::uint64_t maximum_bytes{0};
  std::size_t activity_buffer_bytes{kDefaultActivityBufferBytes};
  std::uint64_t consumer_delay_ms{0};
  std::string frame_buffer;
  std::uint32_t frame_records{0};
  bool output_limit_reached{false};
  std::uint64_t started_timestamp_ns{0};
  std::uint64_t ended_timestamp_ns{0};
  std::uint32_t cupti_version{0};
  bool initialized{false};
  bool defer_start{false};
  std::atomic<bool> activities_enabled{false};
  bool finalized{false};
  // The process that loaded the helper; a fork()ed child inherits the state.
  pid_t owner_pid{0};
  std::string initialization_error;
  std::string driver_version;
  std::string runtime_version;
  std::vector<std::string> requested_activities;
  std::vector<std::string> enabled_activities;
};

// The newest kernel record layout for the CUPTI this file is compiled against.
#if CUPTI_API_VERSION >= 130000
using KernelRecord = CUpti_ActivityKernel10;
#else
using KernelRecord = CUpti_ActivityKernel9;
#endif

// Never destroyed: at exit, a fork()ed child still holds the parent's thread
// objects, whose destructors would call std::terminate.
CaptureState& g_state = *new CaptureState();
std::once_flag g_initialize_once;
std::once_flag g_finalize_once;

// `value` as a quoted JSON string. Quotes, backslashes and control
// characters are escaped; other bytes are copied as they are, so UTF-8 passes
// through unchecked.
std::string JsonString(std::string_view value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20U) {
          constexpr char kHex[] = "0123456789abcdef";
          output << "\\u00" << kHex[(character >> 4U) & 0xFU]
                 << kHex[character & 0xFU];
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  output << '"';
  return output.str();
}

// Whether `value` is well-formed UTF-8, as a strict decoder (Python's) reads
// it: no overlong forms, surrogates or code points past U+10FFFF.
bool IsValidUtf8(std::string_view value) {
  std::size_t index = 0;
  while (index < value.size()) {
    const auto lead = static_cast<unsigned char>(value[index]);
    if (lead < 0x80U) {
      ++index;
      continue;
    }
    std::size_t length = 0;
    std::uint32_t code_point = 0;
    if (lead >= 0xC2U && lead <= 0xDFU) {
      length = 2;
      code_point = lead & 0x1FU;
    } else if (lead >= 0xE0U && lead <= 0xEFU) {
      length = 3;
      code_point = lead & 0x0FU;
    } else if (lead >= 0xF0U && lead <= 0xF4U) {
      length = 4;
      code_point = lead & 0x07U;
    } else {
      return false;
    }
    if (value.size() - index < length) {
      return false;
    }
    for (std::size_t offset = 1; offset < length; ++offset) {
      const auto next = static_cast<unsigned char>(value[index + offset]);
      if ((next & 0xC0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (next & 0x3FU);
    }
    if ((length == 3 && code_point < 0x800U) ||
        (length == 4 && (code_point < 0x10000U || code_point > 0x10FFFFU)) ||
        (code_point >= 0xD800U && code_point <= 0xDFFFU)) {
      return false;
    }
    index += length;
  }
  return true;
}

// CUPTI's message for `result`, or "CUPTI error <code>" when CUPTI has
// none.
std::string CuptiError(CUptiResult result) {
  const char* message = nullptr;
  if (cuptiGetResultString(result, &message) == CUPTI_SUCCESS &&
      message != nullptr) {
    return message;
  }
  return "CUPTI error " + std::to_string(static_cast<int>(result));
}

// strtoull skips leading spaces and accepts a sign, and "-1" wraps to the
// largest value: a setting must start with a digit.
bool StartsWithDigit(const char* value) {
  return value != nullptr && *value >= '0' && *value <= '9';
}

// Parses a decimal setting into `result`. False, with `result` untouched,
// for a null, signed, space-padded, out-of-range, zero or partly numeric
// value.
bool ParsePositiveInteger(const char* value, std::uint64_t* result) {
  if (!StartsWithDigit(value)) {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0ULL) {
    return false;
  }
  *result = static_cast<std::uint64_t>(parsed);
  return true;
}

// As ParsePositiveInteger, but 0 is accepted.
bool ParseNonnegativeInteger(const char* value, std::uint64_t* result) {
  if (!StartsWithDigit(value)) {
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0') {
    return false;
  }
  *result = static_cast<std::uint64_t>(parsed);
  return true;
}

// The comma-separated names in `value`, empty ones skipped; names are not
// trimmed or checked here. Empty for a null `value`.
std::vector<std::string> SplitActivities(const char* value) {
  std::vector<std::string> activities;
  if (value == nullptr) {
    return activities;
  }
  std::string input(value);
  std::size_t start = 0;
  while (start <= input.size()) {
    const std::size_t comma = input.find(',', start);
    const std::size_t end = comma == std::string::npos ? input.size() : comma;
    if (end > start) {
      activities.emplace_back(input.substr(start, end - start));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return activities;
}

// Whether STORMLOG_CUPTI_ACTIVITIES named the activity kind `name`.
bool IsRequested(std::string_view name) {
  for (const std::string& requested : g_state.requested_activities) {
    if (requested == name) {
      return true;
    }
  }
  return false;
}

// Writes all of `content` to `descriptor`, retrying short writes and EINTR.
// False on any other failure, possibly after part of it was written; errno is
// write()'s, and unset when write() wrote nothing without an error.
bool WriteAll(int descriptor, std::string_view content) {
  std::size_t offset = 0;
  while (offset < content.size()) {
    const ssize_t written =
        write(descriptor, content.data() + offset, content.size() - offset);
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      return false;
    }
    offset += static_cast<std::size_t>(written);
  }
  return true;
}

// Records `message` as the capture's error unless one is recorded already:
// the first failure is the one reported. Takes error_mutex.
void SetInitializationError(std::string message) {
  std::lock_guard<std::mutex> lock(g_state.error_mutex);
  if (g_state.initialization_error.empty()) {
    g_state.initialization_error = std::move(message);
  }
}

bool HasInitializationError() {
  std::lock_guard<std::mutex> lock(g_state.error_mutex);
  return !g_state.initialization_error.empty();
}

std::string InitializationError() {
  std::lock_guard<std::mutex> lock(g_state.error_mutex);
  return g_state.initialization_error;
}

void SetSystemError(std::string_view operation) {
  const int saved_errno = errno;
  SetInitializationError(std::string(operation) + ": " +
                         std::strerror(saved_errno));
}

// The version that `symbol_name` reports, from the process's global symbols,
// or else from `library` if the process has already loaded it; empty when
// neither has it. A library opened with RTLD_LOCAL keeps its symbols out of
// the global scope: cudart opens libcuda.so.1 that way, so
// cuDriverGetVersion is never global, and a target can load libcudart the
// same way. RTLD_NOLOAD never loads a library the target did not load.
std::string QueryCudaVersion(const char* symbol_name, const char* library) {
  using VersionFunction = int (*)(int*);
  void* handle = nullptr;
  void* symbol = dlsym(RTLD_DEFAULT, symbol_name);
  if (symbol == nullptr) {
    handle = dlopen(library, RTLD_LAZY | RTLD_NOLOAD);
    if (handle != nullptr) {
      symbol = dlsym(handle, symbol_name);
    }
  }
  int version = 0;
  bool answered = false;
  if (symbol != nullptr) {
    static_assert(sizeof(VersionFunction) == sizeof(symbol));
    VersionFunction version_function = nullptr;
    std::memcpy(&version_function, &symbol, sizeof(version_function));
    answered = version_function(&version) == 0 && version > 0;
  }
  if (handle != nullptr) {
    dlclose(handle);
  }
  return answered ? std::to_string(version) : std::string();
}

std::string QueryDriverVersion() {
  return QueryCudaVersion("cuDriverGetVersion", "libcuda.so.1");
}

// The shared runtime of the CUDA major version the helper is built for; a
// runtime of another major is refused anyway.
std::string QueryRuntimeVersion() {
  const std::string library =
      "libcudart.so." + std::to_string(CUDA_VERSION / 1000);
  return QueryCudaVersion("cudaRuntimeGetVersion", library.c_str());
}

// Appends `value` to `output` as 4 little-endian bytes, the frame header's
// encoding.
void AppendU32(std::string* output, std::uint32_t value) {
  for (int index = 0; index < 4; ++index) {
    output->push_back(static_cast<char>((value >> (index * 8)) & 0xffU));
  }
}

void DropBufferedRecords() {
  g_state.local_dropped_records.fetch_add(g_state.frame_records);
  g_state.bytes_dropped.fetch_add(g_state.frame_buffer.size());
  g_state.frame_buffer.clear();
  g_state.frame_records = 0;
}

// Compresses the buffered records into one frame and appends it to the
// trace; the caller holds output_mutex. A frame that would leave no room for
// the end frame under STORMLOG_CUPTI_MAX_BYTES is dropped, and so is one whose
// length does not fit 32 bits or that cannot be compressed or fully written;
// its records count as dropped and output_limit_reached stops every later
// write. Only a length, compression or write failure is the capture's error.
// A partial write is truncated away, so the trace stays whole frames.
void FlushFrame() {
  if (g_state.frame_buffer.empty()) {
    return;
  }
  const uLong raw_size = static_cast<uLong>(g_state.frame_buffer.size());
  if (raw_size > std::numeric_limits<std::uint32_t>::max()) {
    SetInitializationError("activity frame exceeds 32-bit length");
    DropBufferedRecords();
    g_state.output_limit_reached = true;
    return;
  }
  std::string compressed(compressBound(raw_size), '\0');
  uLongf compressed_size = compressed.size();
  const int result = compress2(
      reinterpret_cast<Bytef*>(compressed.data()), &compressed_size,
      reinterpret_cast<const Bytef*>(g_state.frame_buffer.data()), raw_size,
      kCompressionLevel);
  if (result != Z_OK ||
      compressed_size > std::numeric_limits<std::uint32_t>::max()) {
    SetInitializationError("activity frame compression failed");
    DropBufferedRecords();
    g_state.output_limit_reached = true;
    return;
  }
  compressed.resize(compressed_size);
  std::string frame;
  frame.reserve(kFrameHeaderBytes + compressed.size());
  AppendU32(&frame, static_cast<std::uint32_t>(compressed_size));
  AppendU32(&frame, static_cast<std::uint32_t>(raw_size));
  AppendU32(&frame, g_state.frame_records);
  frame += compressed;
  const std::uint64_t current = g_state.bytes_written.load();
  if (frame.size() > g_state.maximum_bytes - current - kFrameHeaderBytes) {
    g_state.output_limit_reached = true;
    DropBufferedRecords();
    return;
  }
  if (!WriteAll(g_state.trace_fd, frame)) {
    if (ftruncate(g_state.trace_fd, static_cast<off_t>(current)) != 0 ||
        lseek(g_state.trace_fd, static_cast<off_t>(current), SEEK_SET) < 0) {
      SetSystemError("could not restore trace after a partial frame write");
    } else {
      SetInitializationError("could not write a complete trace frame");
    }
    g_state.output_limit_reached = true;
    DropBufferedRecords();
    return;
  }
  g_state.bytes_written.fetch_add(frame.size());
  g_state.uncompressed_bytes_delivered.fetch_add(raw_size);
  g_state.delivered_records.fetch_add(g_state.frame_records);
  g_state.frame_buffer.clear();
  g_state.frame_records = 0;
}

// Appends one JSON line to the frame being built, flushing frames at
// 64 KiB. A record that comes after the trace is closed, or once the output
// limit is reached, is counted as dropped and not written. Takes
// output_mutex.
void WriteRecord(std::string record) {
  record.push_back('\n');
  std::lock_guard<std::mutex> lock(g_state.output_mutex);
  if (g_state.trace_fd < 0) {
    // A buffer delivered after finalization; counted, never written.
    g_state.local_dropped_records.fetch_add(1);
    g_state.bytes_dropped.fetch_add(record.size());
    return;
  }
  if (g_state.output_limit_reached) {
    g_state.local_dropped_records.fetch_add(1);
    g_state.bytes_dropped.fetch_add(record.size());
    return;
  }
  if (g_state.frame_buffer.size() + record.size() > kFrameBufferBytes) {
    FlushFrame();
  }
  if (g_state.output_limit_reached) {
    g_state.local_dropped_records.fetch_add(1);
    g_state.bytes_dropped.fetch_add(record.size());
    return;
  }
  g_state.frame_buffer += record;
  ++g_state.frame_records;
  if (g_state.frame_buffer.size() >= kFrameBufferBytes) {
    FlushFrame();
  }
}

// A record's opening fields, through the clock fields, with no closing
// brace. `start` and `end` are CUPTI timestamps in nanoseconds, written as the
// device interval when `device_interval` and as the CPU interval otherwise; the
// 0/0 pair CUPTI leaves when it has no timing is written as nulls.
std::string RecordPrefix(std::string_view activity_kind,
                         std::string_view record_id, std::uint64_t start,
                         std::uint64_t end, bool device_interval) {
  std::ostringstream output;
  output << "{\"schema_version\":1,\"record_id\":" << JsonString(record_id)
         << ",\"activity_kind\":" << JsonString(activity_kind)
         << ",\"clock_domain\":\"cupti_timestamp_ns\"";
  if (device_interval) {
    output << ",\"cpu_start_ns\":null,\"cpu_end_ns\":null";
    if (start == 0 && end == 0) {
      output << ",\"device_start_ns\":null,\"device_end_ns\":null";
    } else {
      output << ",\"device_start_ns\":" << start
             << ",\"device_end_ns\":" << end;
    }
  } else {
    if (start == 0 && end == 0) {
      output << ",\"cpu_start_ns\":null,\"cpu_end_ns\":null";
    } else {
      output << ",\"cpu_start_ns\":" << start << ",\"cpu_end_ns\":" << end;
    }
    output << ",\"device_start_ns\":null,\"device_end_ns\":null";
  }
  return output.str();
}

// `value` as a JSON decimal string, or null for 0, which CUPTI uses for
// "none".
std::string OptionalIdentifier(std::uint64_t value) {
  return value == 0 ? "null" : JsonString(std::to_string(value));
}

std::string NextRecordId() {
  return "cupti-" + std::to_string(g_state.next_record_id.fetch_add(1));
}

// Writes a runtime or driver API record. Its name is CUPTI's callback name,
// or null when CUPTI cannot name the callback; the importer then names it by
// cbid.
void WriteApiRecord(const CUpti_Activity* record) {
  const auto* activity = reinterpret_cast<const CUpti_ActivityAPI*>(record);
  const bool driver = record->kind == CUPTI_ACTIVITY_KIND_DRIVER;
  const char* kind = driver ? "driver" : "runtime";
  const char* name = nullptr;
  if (cuptiGetCallbackName(driver ? CUPTI_CB_DOMAIN_DRIVER_API
                                  : CUPTI_CB_DOMAIN_RUNTIME_API,
                           activity->cbid, &name) != CUPTI_SUCCESS) {
    name = nullptr;
  }
  std::ostringstream output;
  output << RecordPrefix(kind, NextRecordId(), activity->start, activity->end,
                         false)
         << ",\"correlation_id\":"
         << JsonString(std::to_string(activity->correlationId))
         << ",\"stream_id\":null,\"graph_id\":null,\"graph_node_id\":null"
         << ",\"provenance\":\"cupti_activity\""
         << ",\"uncertainty\":\"CUPTI timestamps; no cross-clock conversion "
            "applied\""
         << ",\"metadata\":{\"cbid\":" << activity->cbid << ",\"name\":"
         << (name == nullptr ? "null" : JsonString(name))
         << ",\"process_id\":" << activity->processId
         << ",\"thread_id\":" << activity->threadId << "}}";
  WriteRecord(output.str());
}

// Writes a kernel record. A kernel CUPTI gave no timestamps is still
// written, with null device times, and counted in kernels_without_timestamps.
void WriteKernelRecord(const CUpti_Activity* record) {
  const auto* activity = reinterpret_cast<const KernelRecord*>(record);
  if (activity->start == 0 && activity->end == 0) {
    // CUPTI could not collect device timing; count it so the loss is visible.
    g_state.kernels_without_timestamps.fetch_add(1);
  }
  std::ostringstream output;
  output << RecordPrefix("kernel", NextRecordId(), activity->start,
                         activity->end, true)
         << ",\"correlation_id\":"
         << JsonString(std::to_string(activity->correlationId))
         << ",\"stream_id\":" << JsonString(std::to_string(activity->streamId))
         << ",\"graph_id\":" << OptionalIdentifier(activity->graphId)
         << ",\"graph_node_id\":" << OptionalIdentifier(activity->graphNodeId)
         << ",\"provenance\":\"cupti_activity\""
         << ",\"uncertainty\":\"CUPTI timestamps; no cross-clock conversion "
            "applied\""
         << ",\"metadata\":{\"name\":"
         << JsonString(activity->name == nullptr ? "" : activity->name)
         << ",\"device_id\":" << activity->deviceId
         << ",\"context_id\":" << activity->contextId << "}}";
  WriteRecord(output.str());
}

void WriteMemcpyRecord(const CUpti_Activity* record) {
  const auto* activity = reinterpret_cast<const CUpti_ActivityMemcpy*>(record);
  std::ostringstream output;
  output << RecordPrefix("memcpy", NextRecordId(), activity->start,
                         activity->end, true)
         << ",\"correlation_id\":"
         << JsonString(std::to_string(activity->correlationId))
         << ",\"stream_id\":" << JsonString(std::to_string(activity->streamId))
         << ",\"graph_id\":null,\"graph_node_id\":null"
         << ",\"provenance\":\"cupti_activity\""
         << ",\"uncertainty\":\"CUPTI timestamps; no cross-clock conversion "
            "applied\""
         << ",\"metadata\":{\"bytes\":" << activity->bytes
         << ",\"copy_kind\":" << static_cast<unsigned int>(activity->copyKind)
         << ",\"device_id\":" << activity->deviceId
         << ",\"context_id\":" << activity->contextId << "}}";
  WriteRecord(output.str());
}

void WriteMemsetRecord(const CUpti_Activity* record) {
  const auto* activity = reinterpret_cast<const CUpti_ActivityMemset*>(record);
  std::ostringstream output;
  output << RecordPrefix("memset", NextRecordId(), activity->start,
                         activity->end, true)
         << ",\"correlation_id\":"
         << JsonString(std::to_string(activity->correlationId))
         << ",\"stream_id\":" << JsonString(std::to_string(activity->streamId))
         << ",\"graph_id\":null,\"graph_node_id\":null"
         << ",\"provenance\":\"cupti_activity\""
         << ",\"uncertainty\":\"CUPTI timestamps; no cross-clock conversion "
            "applied\""
         << ",\"metadata\":{\"bytes\":" << activity->bytes
         << ",\"value\":" << static_cast<unsigned int>(activity->value)
         << ",\"device_id\":" << activity->deviceId
         << ",\"context_id\":" << activity->contextId << "}}";
  WriteRecord(output.str());
}

// Writes a synchronization record, a CPU interval. Its stream is null when
// CUPTI reports its invalid-value sentinel instead of a stream.
void WriteSynchronizationRecord(const CUpti_Activity* record) {
  const auto* activity =
      reinterpret_cast<const CUpti_ActivitySynchronization*>(record);
  std::ostringstream output;
  output << RecordPrefix("synchronization", NextRecordId(), activity->start,
                         activity->end, false)
         << ",\"correlation_id\":"
         << JsonString(std::to_string(activity->correlationId))
         << ",\"stream_id\":"
         // Some CUPTI headers (pip's 13.0.85) define the sentinel as an int.
         << (activity->streamId ==
                     static_cast<std::uint32_t>(CUPTI_SYNCHRONIZATION_INVALID_VALUE)
                 ? "null"
                 : JsonString(std::to_string(activity->streamId)))
         << ",\"graph_id\":null,\"graph_node_id\":null"
         << ",\"provenance\":\"cupti_activity\""
         << ",\"uncertainty\":\"CUPTI timestamps; no cross-clock conversion "
            "applied\""
         << ",\"metadata\":{\"synchronization_type\":"
         << static_cast<unsigned int>(activity->type)
         << ",\"context_id\":" << activity->contextId << "}}";
  WriteRecord(output.str());
}

// Writes one decoded CUPTI record; kinds the helper does not export are
// skipped.
void ProcessActivityRecord(const CUpti_Activity* record) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_DRIVER:
    case CUPTI_ACTIVITY_KIND_RUNTIME:
      WriteApiRecord(record);
      break;
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      WriteKernelRecord(record);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      WriteMemcpyRecord(record);
      break;
    case CUPTI_ACTIVITY_KIND_MEMSET:
      WriteMemsetRecord(record);
      break;
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
      WriteSynchronizationRecord(record);
      break;
    default:
      break;
  }
}

// CUPTI's request for an activity buffer, on CUPTI's threads, which may
// call it concurrently. Allocates STORMLOG_CUPTI_BUFFER_BYTES unless the bytes
// outstanding would pass STORMLOG_CUPTI_MAX_BUFFER_BYTES; the bytes are
// reserved by compare-and-swap, without a lock. A refusal, a null buffer, makes
// CUPTI drop the records it would have held and is counted in buffer_refusals:
// a writer that falls behind costs records, not memory. A failed allocation is
// refused too, and is the capture's error.
void CUPTIAPI BufferRequested(std::uint8_t** buffer, std::size_t* size,
                              std::size_t* maximum_records) {
  const std::uint64_t bytes = g_state.activity_buffer_bytes;
  // Reserve the bytes before allocating: checked and added apart, requests
  // that arrive together could each pass the cap.
  std::uint64_t reserved = g_state.outstanding_buffer_bytes.load();
  do {
    if (reserved + bytes > g_state.maximum_buffer_bytes) {
      // The writer is behind: let CUPTI drop records rather than grow memory.
      g_state.buffer_refusals.fetch_add(1);
      *buffer = nullptr;
      *size = 0;
      *maximum_records = 0;
      return;
    }
  } while (!g_state.outstanding_buffer_bytes.compare_exchange_weak(
      reserved, reserved + bytes));
  void* allocation = nullptr;
  if (posix_memalign(&allocation, alignof(std::max_align_t),
                     g_state.activity_buffer_bytes) != 0) {
    g_state.outstanding_buffer_bytes.fetch_sub(bytes);
    SetInitializationError("could not allocate a CUPTI activity buffer");
    *buffer = nullptr;
    *size = 0;
    *maximum_records = 0;
    return;
  }
  *buffer = static_cast<std::uint8_t*>(allocation);
  *size = g_state.activity_buffer_bytes;
  *maximum_records = 0;
  g_state.buffers_requested.fetch_add(1);
  const std::uint64_t outstanding = reserved + bytes;
  std::uint64_t peak = g_state.peak_outstanding_buffer_bytes.load();
  while (outstanding > peak &&
         !g_state.peak_outstanding_buffer_bytes.compare_exchange_weak(
             peak, outstanding)) {
  }
}

void CUPTIAPI IgnoreCallback(void* /*userdata*/, CUpti_CallbackDomain /*domain*/,
                            CUpti_CallbackId /*callback_id*/,
                            const void* /*callback_data*/) {}

// Decodes and writes one completed buffer, then releases it.
void ConsumeBuffer(std::uint8_t* buffer, std::size_t valid_size) {
  const auto started = std::chrono::steady_clock::now();
  CUpti_Activity* record = nullptr;
  while (valid_size > 0) {
    const CUptiResult result =
        cuptiActivityGetNextRecord(buffer, valid_size, &record);
    if (result == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    }
    if (result != CUPTI_SUCCESS) {
      SetInitializationError("activity decode failed: " + CuptiError(result));
      break;
    }
    if (g_state.discard_records) {
      g_state.discarded_records.fetch_add(1);
    } else {
      ProcessActivityRecord(record);
    }
  }
  if (g_state.consumer_delay_ms > 0) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(g_state.consumer_delay_ms));
  }
  std::free(buffer);
  g_state.outstanding_buffer_bytes.fetch_sub(g_state.activity_buffer_bytes);
  g_state.buffer_completed_ns.fetch_add(static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started)
          .count()));
}

// The writer thread: consumes completed buffers in the order they came until
// StopWriter is called and the queue is empty. Holds queue_mutex only to take
// a buffer, never while decoding or writing it.
void WriterLoop() {
  std::unique_lock<std::mutex> lock(g_state.queue_mutex);
  while (true) {
    g_state.queue_ready.wait(lock, [] {
      return g_state.writer_stopping || !g_state.completed_buffers.empty();
    });
    if (g_state.completed_buffers.empty()) {
      return;
    }
    const auto [buffer, valid_size] = g_state.completed_buffers.front();
    g_state.completed_buffers.pop_front();
    lock.unlock();
    ConsumeBuffer(buffer, valid_size);
    lock.lock();
  }
}

// Drains every queued buffer, then stops the writer thread.
void StopWriter() {
  {
    std::lock_guard<std::mutex> lock(g_state.queue_mutex);
    g_state.writer_stopping = true;
  }
  g_state.queue_ready.notify_one();
  if (g_state.writer_thread.joinable()) {
    g_state.writer_thread.join();
  }
}

// CUPTI's return of a filled buffer: adds CUPTI's drop count, then queues
// the buffer for the writer, so the callback neither decodes nor writes. Once
// StopWriter has run, the buffer is consumed on the calling thread instead,
// and WriteRecord counts its records as dropped once the trace is closed. A
// null buffer, one BufferRequested refused, is ignored.
void CUPTIAPI BufferCompleted(CUcontext context, std::uint32_t stream_id,
                              std::uint8_t* buffer, std::size_t /*size*/,
                              std::size_t valid_size) {
  if (buffer == nullptr) {
    return;
  }
  std::size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped) ==
      CUPTI_SUCCESS) {
    g_state.cupti_dropped_records.fetch_add(dropped);
  }
  {
    std::lock_guard<std::mutex> lock(g_state.queue_mutex);
    if (!g_state.writer_stopping) {
      g_state.completed_buffers.emplace_back(buffer, valid_size);
      g_state.queue_ready.notify_one();
      return;
    }
  }
  // Delivered after finalization: the trace is closed, so every record in it
  // is counted as dropped.
  ConsumeBuffer(buffer, valid_size);
}

// `values` as a JSON array of strings.
std::string JsonStringArray(const std::vector<std::string>& values) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << JsonString(values[index]);
  }
  output << ']';
  return output.str();
}

// Whether the status reached cupti_status.json and the directory was synced
// after the rename; on failure, errno says why.
bool WriteStatus() {
  if (g_state.directory_fd < 0) {
    errno = EBADF;
    return false;
  }
  unlinkat(g_state.directory_fd, kStatusTemporaryFilename, 0);
  const int descriptor =
      openat(g_state.directory_fd, kStatusTemporaryFilename,
             O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (descriptor < 0) {
    return false;
  }
  std::ostringstream output;
  const std::string initialization_error = InitializationError();
  output
      << "{\"schema_version\":1,\"helper_version\":"
      << JsonString(kHelperVersion) << ",\"pid\":" << getpid()
      << ",\"cupti_version\":" << g_state.cupti_version
      << ",\"compiled_cupti_api_version\":" << CUPTI_API_VERSION
      << ",\"compiled_cuda_version\":" << CUDA_VERSION << ",\"driver_version\":"
      << (g_state.driver_version.empty() ? "null"
                                         : JsonString(g_state.driver_version))
      << ",\"runtime_version\":"
      << (g_state.runtime_version.empty() ? "null"
                                          : JsonString(g_state.runtime_version))
      << ",\"started_timestamp_ns\":" << g_state.started_timestamp_ns
      << ",\"ended_timestamp_ns\":" << g_state.ended_timestamp_ns
      << ",\"requested_activities\":"
      << JsonStringArray(g_state.requested_activities)
      << ",\"enabled_activities\":"
      << JsonStringArray(g_state.enabled_activities)
      << ",\"delivered_records\":" << g_state.delivered_records.load()
      << ",\"cupti_dropped_records\":" << g_state.cupti_dropped_records.load()
      << ",\"local_dropped_records\":" << g_state.local_dropped_records.load()
      << ",\"bytes_written\":" << g_state.bytes_written.load()
      << ",\"bytes_dropped\":" << g_state.bytes_dropped.load()
      << ",\"uncompressed_bytes_delivered\":"
      << g_state.uncompressed_bytes_delivered.load()
      << ",\"trace_encoding\":" << JsonString(kTraceEncoding)
      << ",\"zlib_version\":" << JsonString(zlibVersion())
      << ",\"compression_level\":" << kCompressionLevel
      << ",\"output_limit_reached\":"
      << (g_state.output_limit_reached ? "true" : "false")
      << ",\"activity_buffer_bytes\":" << g_state.activity_buffer_bytes
      << ",\"activity_buffers_requested\":" << g_state.buffers_requested.load()
      << ",\"activity_buffer_peak_outstanding_bytes\":"
      << g_state.peak_outstanding_buffer_bytes.load()
      << ",\"buffer_completed_cpu_ns\":" << g_state.buffer_completed_ns.load()
      << ",\"kernels_without_timestamps\":"
      << g_state.kernels_without_timestamps.load()
      << ",\"buffer_refusals\":" << g_state.buffer_refusals.load()
      << ",\"maximum_buffer_bytes\":" << g_state.maximum_buffer_bytes
      << ",\"device_buffer_bytes\":" << g_state.device_buffer_bytes
      << ",\"device_buffer_pool_limit\":" << g_state.device_buffer_pool_limit
      << ",\"flush_period_ms\":" << g_state.flush_period_ms
      << ",\"discard_records\":" << (g_state.discard_records ? "true" : "false")
      << ",\"discarded_records\":" << g_state.discarded_records.load()
      << ",\"defer_start\":" << (g_state.defer_start ? "true" : "false")
      << ",\"activities_enabled\":"
      << (g_state.activities_enabled.load() ? "true" : "false")
      << ",\"consumer_delay_ms\":" << g_state.consumer_delay_ms
      << ",\"maximum_output_bytes\":" << g_state.maximum_bytes
      << ",\"output_directory\":" << JsonString(g_state.output_directory)
      << ",\"finalized\":" << (g_state.finalized ? "true" : "false")
      << ",\"initialization_error\":";
  if (initialization_error.empty()) {
    output << "null";
  } else {
    output << JsonString(initialization_error);
  }
  output << "}\n";
  const std::string content = output.str();
  int failure = 0;
  errno = 0;
  if (!WriteAll(descriptor, content) || fsync(descriptor) != 0) {
    // A write that wrote nothing sets no errno.
    failure = errno == 0 ? EIO : errno;
  }
  if (close(descriptor) != 0 && failure == 0) {
    failure = errno;
  }
  if (failure == 0 &&
      renameat(g_state.directory_fd, kStatusTemporaryFilename,
               g_state.directory_fd, kStatusFilename) != 0) {
    failure = errno;
  }
  if (failure != 0) {
    errno = failure;
    return false;
  }
  // The rename lasts only once the directory is synced: a final status that
  // may be lost cannot vouch for the trace.
  return fsync(g_state.directory_fd) == 0;
}

// Why the capture cannot be trusted against the target's CUDA runtime, or
// empty when it can. A CUPTI from another CUDA major version can lose kernel
// timestamps without reporting a drop (issue #118, protocol v2).
std::string RuntimeMismatch() {
  if (g_state.runtime_version.empty()) {
    g_state.runtime_version = QueryRuntimeVersion();
  }
  if (g_state.runtime_version.empty()) {
    return "CUDA runtime version unavailable; cannot confirm CUPTI matches it";
  }
  if (std::stoi(g_state.runtime_version) / 1000 != CUDA_VERSION / 1000) {
    return "helper built against CUDA " + std::to_string(CUDA_VERSION) +
           " but the target's CUDA runtime is " + g_state.runtime_version +
           "; rebuild it with the target's CUDA major version";
  }
  return {};
}

// Whether EnableActivities enabled the activity kind `name`.
bool IsEnabled(std::string_view name) {
  for (const std::string& enabled : g_state.enabled_activities) {
    if (enabled == name) {
      return true;
    }
  }
  return false;
}

// Disables every requested kind. One that EnableActivities enabled and that
// stays enabled keeps collecting after the stop, so its failure is the
// capture's error; what CUPTI answers for a kind never enabled is not.
void DisableActivities() {
  for (const ActivitySelection& activity : kSupportedActivities) {
    if (!IsRequested(activity.name)) {
      continue;
    }
    const CUptiResult result = cuptiActivityDisable(activity.kind);
    if (result != CUPTI_SUCCESS && IsEnabled(activity.name)) {
      SetInitializationError(std::string("activity ") + activity.name +
                             " could not be disabled: " + CuptiError(result));
    }
  }
}

// Enables every requested activity kind, or none: a kind CUPTI refused would
// be missing from the trace with no drop to show for it. Runs once, and not
// after the capture has failed, since such a capture can never finalize.
bool EnableActivities() {
  if (g_state.activities_enabled.load() || HasInitializationError()) {
    return false;
  }
  // A deferred start runs after the target has used CUDA, so its runtime
  // must be known by now. A start at injection is checked at finalize.
  const std::string mismatch = RuntimeMismatch();
  if (!mismatch.empty() &&
      (g_state.defer_start || !g_state.runtime_version.empty())) {
    SetInitializationError(mismatch);
    return false;
  }
  // Before any kind is enabled, so this failure leaves nothing collecting.
  const CUptiResult timestamp_result =
      cuptiGetTimestamp(&g_state.started_timestamp_ns);
  if (timestamp_result != CUPTI_SUCCESS) {
    SetInitializationError("timestamp initialization failed: " +
                           CuptiError(timestamp_result));
    return false;
  }
  for (const ActivitySelection& activity : kSupportedActivities) {
    if (!IsRequested(activity.name)) {
      continue;
    }
    const CUptiResult result = cuptiActivityEnable(activity.kind);
    if (result != CUPTI_SUCCESS) {
      // The refusal is the error reported; then undo the kinds enabled.
      SetInitializationError(std::string("activity ") + activity.name +
                             " could not be enabled: " + CuptiError(result));
      DisableActivities();
      g_state.enabled_activities.clear();
      return false;
    }
    g_state.enabled_activities.emplace_back(activity.name);
  }
  g_state.activities_enabled.store(true);
  return true;
}

// A child that fork()s without exec and then exits runs the inherited
// atexit handlers. Its descriptors are the parent's, and its writer and
// control threads do not exist, so it must leave the capture alone.
bool InForkedChild() { return getpid() != g_state.owner_pid; }

// Ends the capture, once per process: at exit, or on STOP from the control
// thread. Disables collection, force-flushes CUPTI and writes every buffer,
// then closes the trace. Only a capture that started, matched the target's
// CUDA runtime and wrote everything gets the end frame and the activity.sclz
// name; any other keeps its .partial name. Then writes the final status,
// whose finalized flag is true only for such a capture (a failed status write
// undoes it and is printed to stderr), and removes the control socket. Does
// nothing in a fork()ed child.
void Finalize() {
  if (InForkedChild()) {
    return;
  }
  std::call_once(g_finalize_once, [] {
    if (g_state.initialized) {
      // Disable first so no record is produced after the final flush. A
      // kind that stays enabled fails the capture, so STOP answers ERR and
      // Stormlog says the profiler may still be running.
      DisableActivities();
      const CUptiResult flush_result =
          cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
      if (flush_result != CUPTI_SUCCESS) {
        SetInitializationError("activity flush failed: " +
                               CuptiError(flush_result));
      }
      const CUptiResult timestamp_result =
          cuptiGetTimestamp(&g_state.ended_timestamp_ns);
      if (timestamp_result != CUPTI_SUCCESS) {
        SetInitializationError("final timestamp failed: " +
                               CuptiError(timestamp_result));
      }
      if (g_state.subscriber != nullptr) {
        cuptiUnsubscribe(g_state.subscriber);
        g_state.subscriber = nullptr;
      }
    }
    // The forced flush has queued every buffer; write them before closing.
    StopWriter();
    // Judge the capture before the trace is finalized, so a capture that
    // cannot be trusted keeps its .partial name and no end frame.
    if (g_state.initialized && !g_state.activities_enabled.load()) {
      SetInitializationError("capture was stopped before it was started");
    } else if (g_state.initialized) {
      // The records cannot be trusted unless the runtime matched by now.
      const std::string mismatch = RuntimeMismatch();
      if (!mismatch.empty()) {
        SetInitializationError(mismatch);
      }
    }
    if (g_state.trace_fd >= 0) {
      {
        std::lock_guard<std::mutex> lock(g_state.output_mutex);
        FlushFrame();
        if (!HasInitializationError()) {
          std::string end_frame(kFrameHeaderBytes, '\0');
          if (!WriteAll(g_state.trace_fd, end_frame)) {
            SetInitializationError("could not write trace end frame");
          } else {
            g_state.bytes_written.fetch_add(end_frame.size());
          }
        }
        if (fsync(g_state.trace_fd) != 0) {
          SetSystemError("trace fsync failed");
        }
        if (close(g_state.trace_fd) != 0) {
          SetSystemError("trace close failed");
        }
        g_state.trace_fd = -1;
      }
      if (!HasInitializationError() &&
          renameat(g_state.directory_fd, kTracePartialFilename,
                   g_state.directory_fd, kTraceFilename) != 0) {
        SetSystemError("trace finalization rename failed");
      }
      if (!HasInitializationError() && fsync(g_state.directory_fd) != 0) {
        SetSystemError("capture directory fsync failed");
      }
    }
    g_state.finalized = g_state.initialized && !HasInitializationError();
    // Without a directory there is no status to write, and initialization
    // already printed why.
    if (g_state.directory_fd >= 0 && !WriteStatus()) {
      // Before any allocation, which may set errno.
      const int saved_errno = errno;
      // That status cannot hold the error, so the server's log gets it.
      std::string message =
          std::string("could not write the final cupti_status.json: ") +
          std::strerror(saved_errno);
      if (g_state.finalized) {
        // The importer refuses a trace whose status does not say it
        // finalized, so STOP must not answer OK, and the trace goes back
        // to its .partial name like any other it cannot vouch for.
        g_state.finalized = false;
        SetInitializationError(message);
        if (renameat(g_state.directory_fd, kTraceFilename,
                     g_state.directory_fd, kTracePartialFilename) == 0) {
          fsync(g_state.directory_fd);
        }
      } else {
        // The status left on disk is an earlier one, which may not name
        // the error the capture failed with.
        const std::string error = InitializationError();
        if (!error.empty()) {
          message += "; the capture had already failed: " + error;
        }
      }
      WriteAll(STDERR_FILENO, "stormlog-cupti: " + message + "\n");
    }
    const int control_fd = g_state.control_fd.exchange(-1);
    if (control_fd >= 0) {
      shutdown(control_fd, SHUT_RDWR);
      close(control_fd);
    }
    if (g_state.directory_fd >= 0) {
      unlinkat(g_state.directory_fd, kControlFilename, 0);
      close(g_state.directory_fd);
      g_state.directory_fd = -1;
    }
  });
}

// At exit, waits for the control thread unless it is the caller or this is
// a fork()ed child. Registered before Finalize, so it runs after it, once
// Finalize has shut the socket down and the loop has returned.
void JoinControl() {
  if (InForkedChild()) {
    return;
  }
  if (g_state.control_thread.joinable() &&
      g_state.control_thread.get_id() != std::this_thread::get_id()) {
    g_state.control_thread.join();
  }
}

// MSG_NOSIGNAL: a client that has gone away must not raise SIGPIPE in the
// traced process, whose default action would end it.
void SendReply(int client, std::string_view reply) {
  std::size_t offset = 0;
  while (offset < reply.size()) {
    const ssize_t sent = send(client, reply.data() + offset,
                              reply.size() - offset, MSG_NOSIGNAL);
    if (sent < 0 && errno == EINTR) {
      continue;
    }
    if (sent <= 0) {
      return;
    }
    offset += static_cast<std::size_t>(sent);
  }
}

// The control thread: one 4-byte command per connection on stop.sock. STRT
// enables collection for a deferred start and answers OK, or ERR when the
// start is not deferred, has already run, fails or follows a failure (an
// earlier failed start, for one); the status is rewritten
// either way. STOP finalizes the capture, answers OK only if it finalized
// cleanly, and ends the loop. Anything else, or nothing within 5 s, answers
// ERR. Accept wakes every second, so the loop ends once Finalize closes the
// socket.
void ControlLoop() {
  while (g_state.control_fd >= 0) {
    const int client = accept4(g_state.control_fd, nullptr, nullptr, SOCK_CLOEXEC);
    if (client < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      return;
    }
    timeval timeout{};
    timeout.tv_sec = 5;
    setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    char request[4]{};
    const ssize_t received = recv(client, request, sizeof(request), MSG_WAITALL);
    if (received == static_cast<ssize_t>(sizeof(request)) &&
        std::memcmp(request, "STRT", sizeof(request)) == 0) {
      const bool started = g_state.defer_start && EnableActivities();
      WriteStatus();
      SendReply(client, started ? "OK\n" : "ERR\n");
      close(client);
      continue;
    }
    if (received == static_cast<ssize_t>(sizeof(request)) &&
        std::memcmp(request, "STOP", sizeof(request)) == 0) {
      Finalize();
      SendReply(client, g_state.finalized ? "OK\n" : "ERR\n");
      close(client);
      return;
    }
    SendReply(client, "ERR\n");
    close(client);
  }
}

// Creates the control socket, stop.sock with mode 0600, in the process
// directory, binding it through the directory's /proc/self/fd path, and starts
// the control thread. False, with the capture's error set, on failure;
// Finalize then closes a socket already created.
bool StartControl() {
  g_state.control_fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (g_state.control_fd < 0) {
    SetSystemError("could not create stop socket");
    return false;
  }
  const std::string path = "/proc/self/fd/" +
                           std::to_string(g_state.directory_fd) + "/" +
                           kControlFilename;
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  if (path.size() >= sizeof(address.sun_path)) {
    SetInitializationError("stop socket path is too long");
    return false;
  }
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  if (::bind(g_state.control_fd.load(), reinterpret_cast<sockaddr*>(&address),
           sizeof(address)) != 0 ||
      fchmodat(g_state.directory_fd, kControlFilename, 0600, 0) != 0 ||
      listen(g_state.control_fd, 4) != 0) {
    SetSystemError("could not bind stop socket");
    return false;
  }
  timeval timeout{};
  timeout.tv_sec = 1;
  setsockopt(g_state.control_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
             sizeof(timeout));
  g_state.control_thread = std::thread(ControlLoop);
  return true;
}

// Sets the capture up at injection, once (InitializeInjection). Checks the
// STORMLOG_CUPTI_* settings; creates a private pid-<pid>-XXXXXX directory under
// the output root, which must be this user's with no group or other
// permissions; opens activity.sclz.partial; registers the buffer callbacks;
// subscribes to CUPTI; and, unless the start is deferred, enables the requested
// activities. Then starts the writer and control threads and registers
// Finalize at exit. A failure records the first error and returns with nothing
// collecting; once the process directory is open, the status file says why.
void Initialize() {
  g_state.owner_pid = getpid();
  const char* output_directory = std::getenv(kOutputDirectoryEnv);
  if (output_directory == nullptr || output_directory[0] != '/') {
    SetInitializationError("output directory must be absolute");
    return;
  }
  // The status file names the process directory, and Stormlog reads it as
  // UTF-8, so a capture under any other name could never be imported.
  if (!IsValidUtf8(output_directory)) {
    SetInitializationError("output directory must be valid UTF-8");
    return;
  }
  if (!ParsePositiveInteger(std::getenv(kMaximumBytesEnv),
                            &g_state.maximum_bytes)) {
    SetInitializationError("maximum bytes must be a positive integer");
    return;
  }
  if (g_state.maximum_bytes < sizeof(kTraceMagic) - 1 + kFrameHeaderBytes) {
    SetInitializationError("maximum bytes cannot hold trace framing");
    return;
  }
  if (const char* value = std::getenv(kBufferBytesEnv)) {
    std::uint64_t requested = 0;
    if (!ParsePositiveInteger(value, &requested) || requested < 65536 ||
        requested > 64U * 1024U * 1024U) {
      SetInitializationError("activity buffer bytes must be between 65536 and 67108864");
      return;
    }
    g_state.activity_buffer_bytes = static_cast<std::size_t>(requested);
  }
  if (const char* value = std::getenv(kMaximumBufferBytesEnv)) {
    if (!ParsePositiveInteger(value, &g_state.maximum_buffer_bytes)) {
      SetInitializationError("maximum buffer bytes must be a positive integer");
      return;
    }
  }
  if (g_state.maximum_buffer_bytes < g_state.activity_buffer_bytes) {
    SetInitializationError("maximum buffer bytes must hold one activity buffer");
    return;
  }
  if (const char* value = std::getenv(kConsumerDelayEnv)) {
    if (!ParseNonnegativeInteger(value, &g_state.consumer_delay_ms) ||
        g_state.consumer_delay_ms > 1000) {
      SetInitializationError("consumer delay ms must be between 0 and 1000");
      return;
    }
  }
  if (const char* value = std::getenv(kDeferStartEnv)) {
    if (std::strcmp(value, "1") != 0 && std::strcmp(value, "0") != 0) {
      SetInitializationError("defer start must be 0 or 1");
      return;
    }
    g_state.defer_start = value[0] == '1';
  }
  for (const auto& [name, target] :
       {std::pair<const char*, std::uint64_t*>{kDeviceBufferBytesEnv,
                                               &g_state.device_buffer_bytes},
        {kDeviceBufferPoolLimitEnv, &g_state.device_buffer_pool_limit},
        {kFlushPeriodEnv, &g_state.flush_period_ms}}) {
    if (const char* value = std::getenv(name)) {
      if (!ParseNonnegativeInteger(value, target)) {
        SetInitializationError(std::string(name) + " must be a nonnegative integer");
        return;
      }
    }
  }
  // CUPTI takes the period in 32 bits: a larger one would wrap, and
  // 4294967296 would become 0, which turns periodic flushing off.
  if (g_state.flush_period_ms > std::numeric_limits<std::uint32_t>::max()) {
    SetInitializationError(std::string(kFlushPeriodEnv) +
                           " must be at most 4294967295");
    return;
  }
  if (const char* value = std::getenv(kDiscardEnv)) {
    g_state.discard_records = std::strcmp(value, "1") == 0;
  }
  g_state.requested_activities = SplitActivities(std::getenv(kActivitiesEnv));
  if (g_state.requested_activities.empty()) {
    SetInitializationError("at least one activity must be requested");
    return;
  }
  const int root_fd =
      open(output_directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (root_fd < 0) {
    SetInitializationError("could not open output root");
    return;
  }
  struct stat root_status{};
  if (fstat(root_fd, &root_status) != 0 || root_status.st_uid != geteuid() ||
      (root_status.st_mode & 0077) != 0) {
    close(root_fd);
    SetInitializationError("output root must be owned by this user and private");
    return;
  }
  std::string directory_template =
      std::string(output_directory) + "/pid-" + std::to_string(getpid()) +
      "-XXXXXX";
  std::vector<char> directory_name(directory_template.begin(),
                                   directory_template.end());
  directory_name.push_back('\0');
  if (mkdtemp(directory_name.data()) == nullptr) {
    close(root_fd);
    SetSystemError("could not create private process output directory");
    return;
  }
  close(root_fd);
  g_state.output_directory = directory_name.data();
  g_state.directory_fd = open(g_state.output_directory.c_str(),
                              O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (g_state.directory_fd < 0) {
    SetSystemError("could not open process output directory");
    return;
  }
  g_state.trace_fd =
      openat(g_state.directory_fd, kTracePartialFilename,
             O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (g_state.trace_fd < 0) {
    SetInitializationError("could not create trace output");
    WriteStatus();
    return;
  }
  if (!WriteAll(g_state.trace_fd,
                std::string_view(kTraceMagic, sizeof(kTraceMagic) - 1))) {
    SetInitializationError("could not write trace magic");
    WriteStatus();
    return;
  }
  g_state.bytes_written.store(sizeof(kTraceMagic) - 1);
  const CUptiResult version_result = cuptiGetVersion(&g_state.cupti_version);
  if (version_result != CUPTI_SUCCESS) {
    SetInitializationError("CUPTI version query failed: " +
                           CuptiError(version_result));
    WriteStatus();
    return;
  }
  g_state.driver_version = QueryDriverVersion();
  g_state.runtime_version = QueryRuntimeVersion();
  // The runtime is often loaded after CUDA initializes, when this runs; an
  // unknown version is checked again when capture starts and at finalize.
  if (!g_state.runtime_version.empty()) {
    const std::string mismatch = RuntimeMismatch();
    if (!mismatch.empty()) {
      SetInitializationError(mismatch);
      WriteStatus();
      return;
    }
  }
  CUptiResult result =
      cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);
  if (result != CUPTI_SUCCESS) {
    SetInitializationError("callback registration failed: " +
                           CuptiError(result));
    WriteStatus();
    return;
  }
  for (const auto& [attribute, value] :
       {std::pair<CUpti_ActivityAttribute, std::uint64_t>{
            CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, g_state.device_buffer_bytes},
        {CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
         g_state.device_buffer_pool_limit}}) {
    if (value == 0) {
      continue;
    }
    std::size_t setting = static_cast<std::size_t>(value);
    std::size_t setting_size = sizeof(setting);
    result = cuptiActivitySetAttribute(attribute, &setting_size, &setting);
    if (result != CUPTI_SUCCESS) {
      SetInitializationError("device buffer attribute failed: " +
                             CuptiError(result));
      WriteStatus();
      return;
    }
  }
  if (g_state.flush_period_ms > 0) {
    result = cuptiActivityFlushPeriod(
        static_cast<std::uint32_t>(g_state.flush_period_ms));
    if (result != CUPTI_SUCCESS) {
      SetInitializationError("flush period failed: " + CuptiError(result));
      WriteStatus();
      return;
    }
  }
  for (const std::string& requested : g_state.requested_activities) {
    bool supported = false;
    for (const ActivitySelection& activity : kSupportedActivities) {
      supported = supported || requested == activity.name;
    }
    if (!supported) {
      SetInitializationError("unsupported activity requested: " + requested);
    }
  }
  if (HasInitializationError()) {
    WriteStatus();
    return;
  }
  // CUPTI allows one subscriber per process. Holding it makes a second CUPTI
  // client fail loudly, and finding it held means one is already attached.
  result = cuptiSubscribe(&g_state.subscriber, IgnoreCallback, nullptr);
  if (result != CUPTI_SUCCESS) {
    g_state.subscriber = nullptr;
    SetInitializationError(
        result == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
            ? "another CUPTI client (Nsight Systems, Kineto) holds this process"
            : "CUPTI subscription failed: " + CuptiError(result));
    WriteStatus();
    return;
  }
  if (!g_state.defer_start && !EnableActivities()) {
    cuptiUnsubscribe(g_state.subscriber);
    g_state.subscriber = nullptr;
    WriteStatus();
    return;
  }
  g_state.writer_thread = std::thread(WriterLoop);
  g_state.initialized = true;
  std::atexit(JoinControl);
  std::atexit(Finalize);
  if (!StartControl()) {
    Finalize();
  }
}

// An error found before the process directory exists has no status file to
// go in: print it, so the server's log says why nothing was captured.
void ReportUnrecordedError() {
  if (g_state.directory_fd >= 0 || !HasInitializationError()) {
    return;
  }
  WriteAll(STDERR_FILENO,
           "stormlog-cupti: not capturing: " + InitializationError() + "\n");
}

}  // namespace

#if defined(__GNUC__)
#define STORMLOG_EXPORT __attribute__((visibility("default")))
#else
#define STORMLOG_EXPORT
#endif

// The entry point CUDA calls in a process whose CUDA_INJECTION64_PATH names
// this library. Starts the capture once; an exception is recorded as the
// capture's error, and an error with no status file to go in is printed.
// Always returns 1, so a failed capture never stops the target from
// starting.
extern "C" STORMLOG_EXPORT int InitializeInjection() {
  try {
    std::call_once(g_initialize_once, Initialize);
  } catch (const std::exception& error) {
    SetInitializationError(error.what());
    WriteStatus();
  } catch (...) {
    SetInitializationError("unknown initialization failure");
    WriteStatus();
  }
  ReportUnrecordedError();
  // Tracing failures never prevent the target CUDA application from starting.
  return 1;
}
