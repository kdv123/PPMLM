// Stores symbols and maps them to contiguous integer IDs
// We use the following terms:
//   token  - A string token in the language model (often a single character)
//   symbol - Numeric integer ID for a token

class Vocabulary: CustomStringConvertible
{
    // Use a dictionary to convert strings quickly to numeric IDs.
    // Start with a single entry for the root node.
    private var tokenToSymbol = [Constants.ROOT_TOKEN: Constants.ROOT_SYMBOL]

    // For debug reasons, we may want to go the other way.
    // So let's keep the inverse dictionary as well.
    private var symbolToToken = [Constants.ROOT_SYMBOL: Constants.ROOT_TOKEN]

    // Adds token to the vocabulary and returns its unique numeric symbol ID
    func add(token: String) -> Int
    {
        if let symbol = tokenToSymbol[token]
        {
            // Already exists in the dictionary.
            return symbol
        }
        else
        {
            // Add to dictionary with ID based on current size of vocab.
            let result: Int = tokenToSymbol.count
            tokenToSymbol[token] = result
            symbolToToken[result] = token
            return result
        }
    }
    
    // Add all the characters in a passed in string as a token in the vocab.
    // This allows clients to easily initialize a vocabulary with a single call.
    func addAllCharacters(valid: String)
    {
        for ch in valid
        {
            _ = add(token: String(ch))
        }
    }
    
    // Look up the numeric token ID of a given token.
    // Will return nil if the token isn't in the vocabulary.
    func getSymbol(ofToken: String) -> Int?
    {
        return tokenToSymbol[ofToken]
    }
    
    // Lookup the token string for a given numeric symbol ID.
    func getToken(ofSymbol: Int) -> String?
    {
        return symbolToToken[ofSymbol]
    }
    
    // Find out how many symbols are in this vocabulary
    var size: Int
    {
        return tokenToSymbol.count
    }
    
    // Provide a friendly string version of this instance
    var description: String
    {
        return "Vocabulary (count \(tokenToSymbol.count) \(tokenToSymbol))"
    }
    
}
