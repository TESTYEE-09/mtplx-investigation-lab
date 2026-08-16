import Foundation

public struct MTPLXSettingsStore: Sendable {
    public var settingsURL: URL
    public var encoder: JSONEncoder
    public var decoder: JSONDecoder

    public init(
        settingsURL: URL = MTPLXSettingsStore.defaultSettingsURL(),
        encoder: JSONEncoder = JSONEncoder(),
        decoder: JSONDecoder = JSONDecoder()
    ) {
        self.settingsURL = settingsURL
        self.encoder = encoder
        self.decoder = decoder
        self.encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    }

    public func load() throws -> MTPLXAppConfiguration {
        guard FileManager.default.fileExists(atPath: settingsURL.path) else {
            return MTPLXAppConfiguration()
        }
        let data = try Data(contentsOf: settingsURL)
        return try decoder.decode(MTPLXAppConfiguration.self, from: data)
    }

    public func save(_ configuration: MTPLXAppConfiguration) throws {
        let directory = settingsURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let data = try encoder.encode(configuration)
        try data.write(to: settingsURL, options: [.atomic])
    }

    public static func defaultSettingsURL(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        arguments: [String] = CommandLine.arguments
    ) -> URL {
        if let override = settingsURLOverride(
            environment: environment,
            arguments: arguments
        ) {
            return override
        }
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        return base
            .appendingPathComponent("MTPLX", isDirectory: true)
            .appendingPathComponent("settings.json")
    }

    public static func settingsURLOverride(
        environment: [String: String],
        arguments: [String]
    ) -> URL? {
        if let raw = environment["MTPLX_APP_SETTINGS_PATH"],
           let url = settingsURL(fromOverride: raw) {
            return url
        }

        let names = ["--mtplx-app-settings", "--mtplx-settings-path"]
        for index in arguments.indices {
            let argument = arguments[index]
            for name in names {
                if argument == name {
                    let next = arguments.index(after: index)
                    guard arguments.indices.contains(next) else { continue }
                    if let url = settingsURL(fromOverride: arguments[next]) {
                        return url
                    }
                } else if argument.hasPrefix("\(name)=") {
                    let value = String(argument.dropFirst(name.count + 1))
                    if let url = settingsURL(fromOverride: value) {
                        return url
                    }
                }
            }
        }
        return nil
    }

    private static func settingsURL(fromOverride raw: String) -> URL? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let expanded = (trimmed as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
            .appendingPathComponent(expanded)
    }
}
