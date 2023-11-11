// Represents a single node in the PPM suffix tree data structure

class Node: CustomStringConvertible
{
    private var child: Node?                        // Nodes that extend this node by an additional character
    private var next: Node?
    private var backoff: Node?
    private var count: Int = 0
    private var symbol: Int = Constants.ROOT_ID
    
    // Provide a friendly string version of this instance
    var description: String
    {
        return ""
    }
}
