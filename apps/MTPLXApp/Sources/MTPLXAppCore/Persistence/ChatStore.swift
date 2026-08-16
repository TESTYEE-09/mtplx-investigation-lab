import Foundation
import SwiftData

// MARK: - ChatStore
//
// Factory for the SwiftData `ModelContainer` backing the in-app chat
// surface. Crucially this uses an EXPLICIT `ModelConfiguration` URL
// rather than the SwiftData default, because on macOS the default
// store location/name is shared across apps and can collide with any
// other app that also uses SwiftData. The store lives under
// `~/Library/Application Support/MTPLX/chats.store`.
//
// Each `ChatStore.makeContainer()` call creates a fresh container; the
// app holds one instance for its lifetime (created in `MTPLXApp` and
// passed down via environment). Tests can opt into an in-memory
// container via `makeInMemoryContainer()` to avoid touching disk.

public enum ChatStore {
    /// Subfolder under `~/Library/Application Support/` where MTPLX
    /// persists user data. Reused for the SwiftData store and any
    /// future chat-related files (e.g. exported transcripts).
    public static let appSupportSubdirectory = "MTPLX"
    /// Filename for the SwiftData store. SwiftData will create three
    /// adjacent files: `chats.store`, `chats.store-shm`, `chats.store-wal`.
    public static let storeFilename = "chats.store"
    /// Optional QA/dev override. Production launches should leave this unset.
    public static let storePathEnvironmentVariable = "MTPLX_CHAT_STORE_PATH"
    public static let storePathArgumentNames = [
        "--mtplx-chat-store",
        "--mtplx-chat-store-path"
    ]

    /// Resolves the on-disk URL of the persistent store, creating the
    /// containing directory if it does not exist.
    public static func storeURL() throws -> URL {
        if let explicit = explicitStorePath(
            environment: ProcessInfo.processInfo.environment,
            arguments: CommandLine.arguments
        ) {
            return try explicitStoreURL(explicit)
        }
        let supportRoot = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let mtplxDir = supportRoot.appendingPathComponent(
            appSupportSubdirectory,
            isDirectory: true
        )
        try FileManager.default.createDirectory(
            at: mtplxDir,
            withIntermediateDirectories: true
        )
        return mtplxDir.appendingPathComponent(storeFilename)
    }

    public static func explicitStorePath(
        environment: [String: String],
        arguments: [String]
    ) -> String? {
        if let value = environment[storePathEnvironmentVariable]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !value.isEmpty {
            return value
        }
        for (index, argument) in arguments.enumerated() {
            for name in storePathArgumentNames {
                if argument == name,
                   arguments.indices.contains(index + 1) {
                    let value = arguments[index + 1].trimmingCharacters(in: .whitespacesAndNewlines)
                    return value.isEmpty ? nil : value
                }
                let prefix = name + "="
                if argument.hasPrefix(prefix) {
                    let value = String(argument.dropFirst(prefix.count))
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    return value.isEmpty ? nil : value
                }
            }
        }
        return nil
    }

    public static func explicitStoreURL(_ path: String) throws -> URL {
        let expanded = NSString(string: path).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return url
    }

    /// Build a SwiftData `ModelContainer` for the chat domain.
    /// - Throws: any FileManager / SwiftData error encountered while
    ///   creating the support directory or initializing the container.
    @MainActor
    public static func makeContainer() throws -> ModelContainer {
        let url = try storeURL()
        let schema = Schema([
            ChatConversation.self,
            ChatMessage.self,
            ChatAttachment.self,
            ToolTraceRecord.self,
        ])
        let configuration = ModelConfiguration(
            "MTPLXChats",
            schema: schema,
            url: url,
            allowsSave: true,
            cloudKitDatabase: .none
        )
        return try ModelContainer(for: schema, configurations: [configuration])
    }

    /// In-memory container for tests and previews. Does not touch disk.
    @MainActor
    public static func makeInMemoryContainer() throws -> ModelContainer {
        let schema = Schema([
            ChatConversation.self,
            ChatMessage.self,
            ChatAttachment.self,
            ToolTraceRecord.self,
        ])
        let configuration = ModelConfiguration(
            "MTPLXChatsInMemory",
            schema: schema,
            isStoredInMemoryOnly: true,
            allowsSave: true,
            cloudKitDatabase: .none
        )
        return try ModelContainer(for: schema, configurations: [configuration])
    }
}
