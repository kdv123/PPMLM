// Node in a search tree, which is implemented as a suffix trie that represents
// every suffix of a sequence used during its construction. Please see
// [1] Moffat, Alistair (1990): "Implementing the PPM data compression
//     scheme", IEEE Transactions on Communications, vol. 38, no. 11, pp.
//     1917--1921.
// [2] Esko Ukknonen (1995): "On-line construction of suffix trees",
//     Algorithmica, volume 14, pp. 249--260, Springer, 1995.
// [3] Kennington, C. (2011): "Application of Suffix Trees as an
//     Implementation Technique for Varied-Length N-gram Language Models",
//     MSc. Thesis, Saarland University.

class Node: CustomStringConvertible
{
    static let NULL = UInt32.max
    
    // Node containing the linked list that has symbols extending this node by one more symbol.
    var child: UInt32 = Node.NULL
    
    // Next node in the linked list for seen symbols after our current Node's context.
    private(set) var next: UInt32 = Node.NULL

    // Node in the backoff structure, also known as "vine" structure (see [1]
    // above) and "suffix link" (see [2] above). The backoff for the given node
    // points at the node representing the shorter context. For example, if the
    // current node in the trie represents string "AA" (corresponding to the
    // branch "[R] -> [A] -> [*A*]" in the trie, where [R] stands for root),
    // then its backoff points at the node "A" (represented by "[R] ->
    // [*A*]"). In this case both nodes are in the same branch but they don't
    // need to be. For example, for the node "B" in the trie path for the string
    // "AB" ("[R] -> [A] -> [*B*]") the backoff points at the child node of a
    // different path "[R] -> [*B*]".
    private(set) var backoff: UInt32 = Node.NULL
    
    // Frequency count for this node. Number of times the suffix symbol stored
    // in this node was observed.
    private(set) var count: Int = 1

    // Symbol that this node stores.
    private(set) var symbol: Int = Constants.ROOT_SYMBOL
    
    // Constructor for the default root Node
    init()
    {
    }
    
    // Contruct for a given symbol and possible pointers to other Nodes.
    init(symbol: Int, next: UInt32, backoff: UInt32)
    {
        self.symbol = symbol
        self.next = next
        self.backoff = backoff
    }

    // Add one to the count of this Node.
    func incrementCount()
    {
        count += 1
    }
    
    // Provide a friendly string version of this instance.
    var description: String
    {
        return "(Node symbol \(symbol) " +
        "count \(count) " +
        "child \(child) " +
        "next \(next) " +
        "backoff \(backoff) " +
        ")"
    }
}
