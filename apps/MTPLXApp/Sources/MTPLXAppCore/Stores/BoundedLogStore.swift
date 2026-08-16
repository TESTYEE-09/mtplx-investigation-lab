import Foundation

public struct LogEntry: Equatable, Sendable, Identifiable {
    public enum Stream: String, Sendable {
        case stdout
        case stderr
        case system
    }

    public let id: UUID
    public let date: Date
    public let stream: Stream
    public let message: String

    public init(
        id: UUID = UUID(),
        date: Date = Date(),
        stream: Stream,
        message: String
    ) {
        self.id = id
        self.date = date
        self.stream = stream
        self.message = message
    }
}

public actor BoundedLogStore {
    private let capacity: Int
    private var entries: [LogEntry] = []

    public init(capacity: Int = 500) {
        self.capacity = max(1, capacity)
    }

    public func append(_ message: String, stream: LogEntry.Stream) {
        let normalized = message.trimmingCharacters(in: .newlines)
        guard !normalized.isEmpty else { return }
        entries.append(LogEntry(stream: stream, message: normalized))
        if entries.count > capacity {
            entries.removeFirst(entries.count - capacity)
        }
    }

    public func snapshot() -> [LogEntry] {
        entries
    }

    public func clear() {
        entries.removeAll(keepingCapacity: true)
    }
}
