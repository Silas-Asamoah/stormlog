// Opt-in CUPTI Activity injection for bounded Stormlog trace sidecars.

#include <cupti.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr char kHelperVersion[] = "0.1.0";
constexpr char kOutputDirectoryEnv[] = "STORMLOG_CUPTI_OUTPUT_DIR";
constexpr char kMaximumBytesEnv[] = "STORMLOG_CUPTI_MAX_BYTES";
constexpr char kActivitiesEnv[] = "STORMLOG_CUPTI_ACTIVITIES";
constexpr char kTracePartialFilename[] = "activity.ndjson.partial";
constexpr char kTraceFilename[] = "activity.ndjson";
constexpr char kStatusFilename[] = "cupti_status.json";
constexpr char kStatusTemporaryFilename[] = "cupti_status.json.tmp";
constexpr std::size_t kActivityBufferBytes = 4U * 1024U * 1024U;

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
  int directory_fd{-1};
  int trace_fd{-1};
  std::uint64_t maximum_bytes{0};
  std::uint64_t started_timestamp_ns{0};
  std::uint64_t ended_timestamp_ns{0};
  std::uint32_t cupti_version{0};
  bool initialized{false};
  bool finalized{false};
  std::string initialization_error;
  std::string driver_version;
  std::string runtime_version;
  std::vector<std::string> requested_activities;
  std::vector<std::string> enabled_activities;
};

CaptureState g_state;
std::once_flag g_initialize_once;
std::once_flag g_finalize_once;

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

std::string CuptiError(CUptiResult result) {
  const char* message = nullptr;
  if (cuptiGetResultString(result, &message) == CUPTI_SUCCESS &&
      message != nullptr) {
    return message;
  }
  return "CUPTI error " + std::to_string(static_cast<int>(result));
}

bool ParsePositiveInteger(const char* value, std::uint64_t* result) {
  if (value == nullptr || *value == '\0') {
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

bool IsRequested(std::string_view name) {
  for (const std::string& requested : g_state.requested_activities) {
    if (requested == name) {
      return true;
    }
  }
  return false;
}

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

std::string QueryCudaVersion(const char* symbol_name) {
  using VersionFunction = int (*)(int*);
  void* symbol = dlsym(RTLD_DEFAULT, symbol_name);
  if (symbol == nullptr) {
    return {};
  }
  static_assert(sizeof(VersionFunction) == sizeof(symbol));
  VersionFunction version_function = nullptr;
  std::memcpy(&version_function, &symbol, sizeof(version_function));
  int version = 0;
  if (version_function(&version) != 0 || version <= 0) {
    return {};
  }
  return std::to_string(version);
}

void WriteRecord(std::string record) {
  record.push_back('\n');
  std::lock_guard<std::mutex> lock(g_state.output_mutex);
  const std::uint64_t current = g_state.bytes_written.load();
  if (current >= g_state.maximum_bytes ||
      record.size() > g_state.maximum_bytes - current) {
    g_state.local_dropped_records.fetch_add(1);
    g_state.bytes_dropped.fetch_add(record.size());
    return;
  }
  if (!WriteAll(g_state.trace_fd, record)) {
    if (ftruncate(g_state.trace_fd, static_cast<off_t>(current)) != 0 ||
        lseek(g_state.trace_fd, static_cast<off_t>(current), SEEK_SET) < 0) {
      SetSystemError("could not restore trace after a partial write");
    } else {
      SetInitializationError("could not write a complete trace record");
    }
    g_state.local_dropped_records.fetch_add(1);
    g_state.bytes_dropped.fetch_add(record.size());
    return;
  }
  g_state.bytes_written.fetch_add(record.size());
  g_state.delivered_records.fetch_add(1);
}

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

std::string OptionalIdentifier(std::uint64_t value) {
  return value == 0 ? "null" : JsonString(std::to_string(value));
}

std::string NextRecordId() {
  return "cupti-" + std::to_string(g_state.next_record_id.fetch_add(1));
}

void WriteApiRecord(const CUpti_Activity* record) {
  const auto* activity = reinterpret_cast<const CUpti_ActivityAPI*>(record);
  const char* kind =
      record->kind == CUPTI_ACTIVITY_KIND_DRIVER ? "driver" : "runtime";
  std::ostringstream output;
  output << RecordPrefix(kind, NextRecordId(), activity->start, activity->end,
                         false)
         << ",\"correlation_id\":"
         << JsonString(std::to_string(activity->correlationId))
         << ",\"stream_id\":null,\"graph_id\":null,\"graph_node_id\":null"
         << ",\"provenance\":\"cupti_activity\""
         << ",\"uncertainty\":\"CUPTI timestamps; no cross-clock conversion "
            "applied\""
         << ",\"metadata\":{\"cbid\":" << activity->cbid
         << ",\"process_id\":" << activity->processId
         << ",\"thread_id\":" << activity->threadId << "}}";
  WriteRecord(output.str());
}

void WriteKernelRecord(const CUpti_Activity* record) {
  const auto* activity = reinterpret_cast<const CUpti_ActivityKernel9*>(record);
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

void WriteSynchronizationRecord(const CUpti_Activity* record) {
  const auto* activity =
      reinterpret_cast<const CUpti_ActivitySynchronization*>(record);
  std::ostringstream output;
  output << RecordPrefix("synchronization", NextRecordId(), activity->start,
                         activity->end, false)
         << ",\"correlation_id\":"
         << JsonString(std::to_string(activity->correlationId))
         << ",\"stream_id\":"
         << (activity->streamId == CUPTI_SYNCHRONIZATION_INVALID_VALUE
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

void CUPTIAPI BufferRequested(std::uint8_t** buffer, std::size_t* size,
                              std::size_t* maximum_records) {
  void* allocation = nullptr;
  if (posix_memalign(&allocation, alignof(std::max_align_t),
                     kActivityBufferBytes) != 0) {
    SetInitializationError("could not allocate a CUPTI activity buffer");
    *buffer = nullptr;
    *size = 0;
    *maximum_records = 0;
    return;
  }
  *buffer = static_cast<std::uint8_t*>(allocation);
  *size = kActivityBufferBytes;
  *maximum_records = 0;
}

void CUPTIAPI BufferCompleted(CUcontext context, std::uint32_t stream_id,
                              std::uint8_t* buffer, std::size_t /*size*/,
                              std::size_t valid_size) {
  if (buffer == nullptr) {
    return;
  }
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
    ProcessActivityRecord(record);
  }
  std::size_t dropped = 0;
  if (cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped) ==
      CUPTI_SUCCESS) {
    g_state.cupti_dropped_records.fetch_add(dropped);
  }
  std::free(buffer);
}

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

void WriteStatus() {
  if (g_state.directory_fd < 0) {
    return;
  }
  unlinkat(g_state.directory_fd, kStatusTemporaryFilename, 0);
  const int descriptor =
      openat(g_state.directory_fd, kStatusTemporaryFilename,
             O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (descriptor < 0) {
    return;
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
      << ",\"finalized\":" << (g_state.finalized ? "true" : "false")
      << ",\"initialization_error\":";
  if (initialization_error.empty()) {
    output << "null";
  } else {
    output << JsonString(initialization_error);
  }
  output << "}\n";
  const std::string content = output.str();
  bool persisted = WriteAll(descriptor, content);
  if (persisted && fsync(descriptor) != 0) {
    persisted = false;
  }
  if (close(descriptor) != 0) {
    persisted = false;
  }
  if (persisted && renameat(g_state.directory_fd, kStatusTemporaryFilename,
                            g_state.directory_fd, kStatusFilename) != 0) {
    persisted = false;
  }
  if (persisted) {
    fsync(g_state.directory_fd);
  }
}

void Finalize() {
  std::call_once(g_finalize_once, [] {
    if (g_state.initialized) {
      const CUptiResult flush_result =
          cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
      if (flush_result != CUPTI_SUCCESS) {
        SetInitializationError("activity flush failed: " +
                               CuptiError(flush_result));
      }
      for (const ActivitySelection& activity : kSupportedActivities) {
        if (IsRequested(activity.name)) {
          cuptiActivityDisable(activity.kind);
        }
      }
      const CUptiResult timestamp_result =
          cuptiGetTimestamp(&g_state.ended_timestamp_ns);
      if (timestamp_result != CUPTI_SUCCESS) {
        SetInitializationError("final timestamp failed: " +
                               CuptiError(timestamp_result));
      }
    }
    if (g_state.trace_fd >= 0) {
      if (fsync(g_state.trace_fd) != 0) {
        SetSystemError("trace fsync failed");
      }
      if (close(g_state.trace_fd) != 0) {
        SetSystemError("trace close failed");
      }
      g_state.trace_fd = -1;
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
    WriteStatus();
    if (g_state.directory_fd >= 0) {
      close(g_state.directory_fd);
      g_state.directory_fd = -1;
    }
  });
}

void Initialize() {
  const char* output_directory = std::getenv(kOutputDirectoryEnv);
  if (output_directory == nullptr || output_directory[0] != '/') {
    SetInitializationError("output directory must be absolute");
    return;
  }
  if (!ParsePositiveInteger(std::getenv(kMaximumBytesEnv),
                            &g_state.maximum_bytes)) {
    SetInitializationError("maximum bytes must be a positive integer");
    return;
  }
  g_state.requested_activities = SplitActivities(std::getenv(kActivitiesEnv));
  if (g_state.requested_activities.empty()) {
    SetInitializationError("at least one activity must be requested");
    return;
  }
  g_state.directory_fd =
      open(output_directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (g_state.directory_fd < 0) {
    SetInitializationError("could not open output directory");
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
  const CUptiResult version_result = cuptiGetVersion(&g_state.cupti_version);
  if (version_result != CUPTI_SUCCESS) {
    SetInitializationError("CUPTI version query failed: " +
                           CuptiError(version_result));
    WriteStatus();
    return;
  }
  g_state.driver_version = QueryCudaVersion("cuDriverGetVersion");
  g_state.runtime_version = QueryCudaVersion("cudaRuntimeGetVersion");
  CUptiResult result =
      cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);
  if (result != CUPTI_SUCCESS) {
    SetInitializationError("callback registration failed: " +
                           CuptiError(result));
    WriteStatus();
    return;
  }
  for (const ActivitySelection& activity : kSupportedActivities) {
    if (!IsRequested(activity.name)) {
      continue;
    }
    result = cuptiActivityEnable(activity.kind);
    if (result == CUPTI_SUCCESS) {
      g_state.enabled_activities.emplace_back(activity.name);
    }
  }
  if (g_state.enabled_activities.empty()) {
    SetInitializationError("none of the requested activities could be enabled");
    WriteStatus();
    return;
  }
  result = cuptiGetTimestamp(&g_state.started_timestamp_ns);
  if (result != CUPTI_SUCCESS) {
    SetInitializationError("timestamp initialization failed: " +
                           CuptiError(result));
    WriteStatus();
    return;
  }
  g_state.initialized = true;
  std::atexit(Finalize);
}

}  // namespace

#if defined(__GNUC__)
#define STORMLOG_EXPORT __attribute__((visibility("default")))
#else
#define STORMLOG_EXPORT
#endif

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
  // Tracing failures never prevent the target CUDA application from starting.
  return 1;
}
