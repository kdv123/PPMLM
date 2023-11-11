// Stores symbols and maps them to contiguous integer IDs

class Vocabulary: CustomStringConvertible
{
    // Use a dictionary to convert strings quickly to numeric IDs.
    // Start with a single entry for the root node.
    private var symbolToID = [Constants.ROOT_NAME: Constants.ROOT_ID]
    
    // Adds symbol to the vocabulary and returns its unique ID
    func add(symbol: String) -> Int
    {
        if let ID = symbolToID[symbol]
        {
            // Already exists in the dictionary.
            return ID
        }
        else
        {
            // Add to dictionary with ID based on current size of vocab.
            let result: Int = symbolToID.count
            symbolToID[symbol] = result
            return result
        }
    }
    
    // Find out how many symbols are in this vocabulary
    var size: Int
    {
        return symbolToID.count
    }
    
    // Provide a friendly string version of this instance
    var description: String
    {
        return "Vocabulary (count \(symbolToID.count) \(symbolToID))"
    }
    
}
