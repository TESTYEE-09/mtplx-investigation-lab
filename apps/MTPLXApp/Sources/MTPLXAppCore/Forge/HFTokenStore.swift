import Foundation
import Security

// MARK: - HFTokenStore
//
// Keychain wrapper for the user's Hugging Face write token. Stores
// the token under a stable service id so the same Keychain item
// survives app updates. On first launch, opportunistically imports
// from the standard `huggingface-cli login` token file at
// `~/.cache/huggingface/token` so users who've already authenticated
// via the CLI don't have to paste again.
//
// We deliberately do NOT integrate the official `huggingface_hub`
// OAuth device flow — token-paste is the dominant pattern across
// HF's own ecosystem (huggingface_hub.HfApi, mlx-lm.convert,
// jjang-ai/jangq) and OAuth would add an engineering bill for an
// audience that's predominantly developer-tier.

public struct HFTokenStore: Sendable {
    public static let service = "com.youssofal.mtplx.app.hf-token"
    public static let account = "default"

    public init() {}

    // MARK: Read

    /// Returns the saved Keychain token if present; otherwise tries
    /// to import from `~/.cache/huggingface/token` on first call.
    public func load() -> String? {
        if let keychainToken = readKeychain() { return keychainToken }
        if let imported = readCLITokenFile(), !imported.isEmpty {
            _ = save(imported)
            return imported
        }
        return nil
    }

    // MARK: Write

    @discardableResult
    public func save(_ token: String) -> Bool {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        let data = Data(trimmed.utf8)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.service,
            kSecAttrAccount as String: Self.account
        ]
        SecItemDelete(query as CFDictionary)
        var attrs = query
        attrs[kSecValueData as String] = data
        attrs[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(attrs as CFDictionary, nil) == errSecSuccess
    }

    public func delete() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.service,
            kSecAttrAccount as String: Self.account
        ]
        SecItemDelete(query as CFDictionary)
    }

    public var hasToken: Bool {
        readKeychain() != nil || readCLITokenFile() != nil
    }

    // MARK: Internals

    private func readKeychain() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.service,
            kSecAttrAccount as String: Self.account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess, let data = result as? Data, let token = String(data: data, encoding: .utf8) else {
            return nil
        }
        return token.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func readCLITokenFile() -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let candidates = [
            home.appendingPathComponent(".cache/huggingface/token"),
            home.appendingPathComponent(".cache/huggingface/stored_tokens").appendingPathComponent("default")
        ]
        for url in candidates {
            if let raw = try? String(contentsOf: url, encoding: .utf8) {
                let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
        }
        return nil
    }
}
