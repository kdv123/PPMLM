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


struct Node: CustomStringConvertible
{
    // Constant index value used to denote we aren't pointing anywhere.
    static let NULL = UInt32.max

    // Memory consumed by the struct seems to align on a multiple of 4 bytes.
    // So to reduce memory consumption, we want all our attributes to be as small as possible.
    // We provide getters and setters to convert these to Int type for use by the client.
    // Note that this seems to slow things down a bit, but avoids messy casting in the client.
    
    // Index of the Node containing the linked list that has symbols extending this node by one more symbol.
    private var _child: UInt32 = Node.NULL
    var child: Int
    {
        get 
        {
            return Int(_child)
        }
        set
        {
            assert(newValue <= UInt32.max, "Index exceeded maximum value!")
            _child = UInt32(newValue)
        }
    }
    
    // Index of next node in the linked list for seen symbols after our current Node's context.
    private var _next: UInt32 = Node.NULL
    var next: Int
    {
        get
        {
            return Int(_next)
        }
        set
        {
            assert(newValue <= UInt32.max, "Index exceeded maximum value!")
            _next = UInt32(newValue)
        }
    }
    
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
    private var _backoff: UInt32 = Node.NULL
    var backoff: Int
    {
        get
        {
            return Int(_backoff)
        }
        set
        {
            assert(newValue <= UInt32.max, "Index exceeded maximum value!")
            _backoff = UInt32(newValue)
        }
    }
    
    // Frequency count for this node. Number of times the suffix symbol stored
    // in this node was observed.
    private var _count: UInt16 = 1
    var count: Int
    {
        get
        {
            return Int(_count)
        }
        set
        {
            assert(newValue <= UInt16.max, "Count exceeded maximum value!")
            _count = UInt16(newValue)
        }
    
    }
    
    // Symbol that this node stores.
    private var _symbol: UInt16 = UInt16(Constants.ROOT_SYMBOL)
    var symbol: Int
    {
        get
        {
            return Int(_symbol)
        }
        set
        {
            assert(newValue <= UInt16.max, "Symbol exceeded maximum value!")
            _symbol = UInt16(newValue)
        }
    }
    
    // Constructor for the default root Node
    init()
    {
    }
    
    // Contruct for a given symbol and possible pointers to other Nodes.
    init(symbol: Int, next: Int, backoff: Int)
    {
        self.symbol = symbol
        self.next = next
        self.backoff = backoff
    }

    // Add one to the count of this Node.
    mutating func incrementCount()
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
