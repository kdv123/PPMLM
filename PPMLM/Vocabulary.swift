// Stores symbols and maps them to contiguous integer IDs
// We use the following terms:
//   token  - A string token in the language model (often a single character)
//   symbol - Numeric integer ID for a token

class Vocabulary: CustomStringConvertible
{
    // Use a dictionary to convert strings quickly to numeric IDs.
    // Start with a single entry for the root node.
    private var tokenToSymbol = [Constants.ROOT_TOKEN: Constants.ROOT_SYMBOL]
    
    // Adds symbol to the vocabulary and returns its unique ID
    func add(symbol: String) -> Int
    {
        if let symbol = tokenToSymbol[symbol]
        {
            // Already exists in the dictionary.
            return symbol
        }
        else
        {
            // Add to dictionary with ID based on current size of vocab.
            let result: Int = tokenToSymbol.count
            tokenToSymbol[symbol] = result
            return result
        }
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
