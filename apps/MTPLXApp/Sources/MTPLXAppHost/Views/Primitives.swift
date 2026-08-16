// MARK: - Primitives (migration shim)
//
// The 11 types this file used to host have been split into 10 sibling
// files under `Views/Common/` per the swiftui-pro views.md rule
// ("Each type … should be in its own Swift file"). The new homes are:
//
//   Card                  → Views/Common/Card.swift
//   StatTile              → Views/Common/StatTile.swift
//   MicroHeader           → Views/Common/MicroHeader.swift
//   FormRow / FormToggleRow → Views/Common/FormRow.swift
//   MetricRow             → Views/Common/MetricRow.swift
//   PillBadge             → Views/Common/PillBadge.swift
//   EmptyStateView        → Views/Common/EmptyStateView.swift
//   HBarRow               → Views/Common/HBarRow.swift
//   StackedBarSegment / StackedBar → Views/Common/StackedBar.swift
//   FlowLayout            → Views/Common/FlowLayout.swift
//
// File is preserved (instead of deleted) per RULES.md: "Never delete
// files. 4 TB SSD. Move to a new directory or rename. Never `rm`."
