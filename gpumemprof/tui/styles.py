"""Styles shared by the Textual TUI app."""

from __future__ import annotations

TUI_APP_CSS = """
    TabbedContent {
        padding: 1;
    }

    RichLog {
        height: 1fr;
        border: solid gray;
    }

    Button {
        margin: 0 1 1 0;
        height: 5;
        width: auto;
        min-width: 16;
        max-width: 30;
        padding: 1 3;
        content-align: center middle;
        text-style: bold;
        color: #ffffff;
        background: $panel;
    }

    Button.-primary {
        color: #ffffff;
        background: $primary;
        border: solid $primary-lighten-1;
    }

    Button.-success {
        color: #ffffff;
        background: $success;
        border: solid $success-lighten-1;
    }

    Button.-warning {
        color: #000000;
        background: $warning;
        border: solid $warning-lighten-1;
    }

    Button.-error {
        color: #ffffff;
        background: $error;
        border: solid $error-lighten-1;
    }

    Button:hover {
        opacity: 0.9;
    }

    #table-pytorch,
    #table-tensorflow {
        height: 12;
        border: solid gray;
    }

    #pytorch-tab,
    #tensorflow-tab {
        layout: vertical;
    }

    #cli-tab {
        layout: vertical;
        height: 1fr;
        border: solid gray;
        padding: 0 1;
    }

    #cli-buttons-row1,
    #cli-buttons-row2 {
        layout: horizontal;
        content-align: left middle;
        height: auto;
        min-height: 6;
    }

    #cli-buttons-row2 {
        margin-top: 0;
        margin-bottom: 1;
    }

    #cli-runner {
        layout: horizontal;
        content-align: left middle;
        margin: 1 0;
    }

    #cli-command-input {
        width: 1fr;
        padding: 0 1;
        height: 5;
    }

    #cli-loader {
        height: 3;
    }

    #monitoring-tab {
        layout: vertical;
        height: 1fr;
        border: solid gray;
        padding: 0 1;
    }

    #monitor-status {
        margin-bottom: 1;
    }

    #monitor-controls-row1,
    #monitor-controls-row2,
    #monitor-controls-row3 {
        layout: horizontal;
        content-align: left middle;
        margin-bottom: 1;
        height: auto;
        min-height: 6;
    }

    #monitor-thresholds {
        layout: horizontal;
        content-align: left middle;
        margin-bottom: 1;
    }

    #monitor-thresholds Label {
        width: 12;
    }

    #monitor-thresholds Input {
        width: 12;
        margin-right: 1;
    }

    #monitor-alerts-table {
        height: 8;
        border: solid gray;
        margin-top: 1;
    }

    #monitor-stats {
        height: 10;
        border: solid gray;
    }

    #monitor-log {
        height: 1fr;
        border: solid gray;
        margin-top: 1;
    }

    #visualizations-tab {
        layout: vertical;
        height: 1fr;
        border: solid gray;
        padding: 0 1;
    }

    #visual-buttons {
        layout: horizontal;
        content-align: left middle;
        margin-bottom: 1;
        height: auto;
        min-height: 6;
        overflow: hidden;
    }

    #timeline-stats {
        height: 8;
        border: solid gray;
        margin-bottom: 1;
    }

    #timeline-canvas {
        border: solid gray;
        padding: 1;
    }

    #visual-log {
        height: 8;
        border: solid gray;
        margin-top: 1;
    }

    #overview-welcome {
        border: round $primary;
        padding: 1;
        margin: 0 0 1 0;
        background: $panel;
        text-align: center;
        text-style: bold;
        color: $accent;
        min-height: 10;
        content-align: center middle;
    }

    #welcome-info {
        border: solid $primary;
        padding: 2;
        margin: 0 0 1 0;
        background: $surface;
        height: auto;
        min-height: 15;
    }

    #welcome-info Markdown {
        color: $text;
    }

    #pytorch-profile-controls,
    #tensorflow-profile-controls {
        layout: horizontal;
        content-align: left middle;
        margin-top: 1;
        height: auto;
        min-height: 6;
    }

    #pytorch-profile-table,
    #tensorflow-profile-table {
        height: 12;
        border: solid gray;
        margin-top: 1;
    }
"""
