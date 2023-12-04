// Stores symbols and maps them to contiguous integer IDs
// We use the following terms:
//   token  - A string token in the language model (often a single character)
//   symbol - Numeric integer ID for a token

struct Vocabulary: CustomStringConvertible
{
    // Use a dictionary to convert strings quickly to numeric IDs.
    // Start with a single entry for the root node.
    private var tokenToSymbol = [Constants.ROOT_TOKEN: PPMLanguageModel.Symbol(Constants.ROOT_SYMBOL)]

    // For debug reasons, we may want to go the other way.
    // So let's keep the inverse dictionary as well.
    private var symbolToToken = [PPMLanguageModel.Symbol(Constants.ROOT_SYMBOL): Constants.ROOT_TOKEN]

    // Adds token to the vocabulary and returns its unique numeric symbol ID
    mutating func add(token: String) -> PPMLanguageModel.Symbol
    {
        if let symbol = tokenToSymbol[token]
        {
            // Already exists in the dictionary.
            return symbol
        }
        else
        {
            // Add to dictionary with ID based on current size of vocab.
            assert(tokenToSymbol.count < PPMLanguageModel.Symbol.max, "Hit limit of size of vocab data type!")
            let result = PPMLanguageModel.Symbol(tokenToSymbol.count)
            tokenToSymbol[token] = result
            symbolToToken[result] = token
            return result
        }
    }
    
    // Add all the characters in a passed in string as a token in the vocab.
    // This allows clients to easily initialize a vocabulary with a single call.
    mutating func addAllCharacters(valid: String)
    {
        for ch in valid
        {
            _ = add(token: String(ch))
        }
    }
    
    // Look up the numeric token ID of a given token.
    // Will return nil if the token isn't in the vocabulary.
    func getSymbol(ofToken: String) -> PPMLanguageModel.Symbol?
    {
        return tokenToSymbol[ofToken]
    }
    
    // Lookup the token string for a given numeric symbol ID.
    func getToken(ofSymbol: PPMLanguageModel.Symbol) -> String?
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
