// Prediction by partial matching (PPM) language model

class PPMLanguageModel: CustomStringConvertible 
{
    private let vocab: Vocabulary
    private let root: Node
    private let rootContext: Context
    private let maxOrder: Int
    private var numNodes: Int
    var useExclusion: Bool = false
    
    init(vocab: Vocabulary, maxOrder: Int)
    {
        self.vocab = vocab
        assert(vocab.size > 1, "Expecting at least two symbols in the vocabulary")
        
        self.maxOrder = maxOrder
        self.root = Node()
        self.rootContext = Context(order: 0, head: self.root)
        self.numNodes = 1
    }
    
    // Add the specified symbol to the specified Node.
    // Returns Node of the existing or the newly created Node.
    private func add(symbol: Int, toNode: Node) -> Node
    {
        if let symbolNode = toNode.findChildWith(symbol: symbol)
        {
            // Update the counts for the given node.  Only updates the counts for
            // the highest order already existing node for the symbol ('single
            // counting' or 'update exclusion').
            symbolNode.incrementCount()
            return symbolNode
        }
        else
        {
            // Symbol does not exist under the given node. Create a new child node
            // and update the backoff structure for lower contexts.

            // Figure out the backoff Node to use for creating the new Node.
            var backoff = root  // Shortest possible context.
            if toNode !== root
            {
                if let toNodeBackoff = toNode.backoff
                {
                    backoff = add(symbol: symbol, toNode: toNodeBackoff)
                }
                else
                {
                    assertionFailure("Expected valid backoff node")
                }
            }
            
            var symbolNode = Node(symbol: symbol, next: toNode.child, backoff: backoff)
            toNode.child = symbolNode
            numNodes += 1
            return symbolNode
        }
    }
        
    // Creates new context which is initially empty.
    func createEmptyContext() -> Context
    {
        return Context(order: rootContext.order, head: rootContext.head)
    }
    
    // Clones existing context.
    func clone(context: Context) -> Context
    {
        return Context(order: context.order, head: context.head)
    }

    // Adds symbol to the supplied context. Does not update the model.
    func add(symbol: Int, toContext: Context)
    {
        // Only add valid symbols.
        if symbol <= Constants.ROOT_SYMBOL
        {
            return
        }
        assert(symbol < vocab.size, "Invalid symbol: \(symbol)")
            
        while true
        {
            if toContext.order < maxOrder
            {
                // Extend the current context.
                let childNode = toContext.head.findChildWith(symbol: symbol)
                if let childNodeUnwrapped = childNode
                {
                    toContext.head = childNodeUnwrapped
                    toContext.order += 1
                    return
                }
            }
            // Try to extend the shorter context.
            toContext.order -= 1
            if let backoff = toContext.head.backoff
            {
                toContext.head = backoff
            }
            else
            {
                break
            }
        }
        // TODO: Is this correct without the if? line 267 in pppm_language_model.js
        toContext.head = root
        toContext.order = 0
    }

    // Adds symbol to the supplied context and update the model.
    func addAndUpdate(symbol: Int, toContext: Context)
    {
        // Only add valid symbols.
        if symbol <= Constants.ROOT_SYMBOL
        {
            return
        }
        assert(symbol < vocab.size, "Invalid symbol: \(symbol)")
        
        let symbolNode = add(symbol: symbol, toNode: toContext.head)
        // TODO: Is this needed?
        assert(symbolNode === toContext.head.findChildWith(symbol: symbol), "failed to find added child")

        toContext.head = symbolNode
        toContext.order += 1
        
        // TODO: Do we really need a loop here? Can't we only go over by 1?
        while toContext.order > maxOrder
        {
            if let backoff = toContext.head.backoff
            {
                toContext.head = backoff
                toContext.order -= 1
            }
            else
            {
                assertionFailure("backoff was nil while shortening after add")
            }
        }
    }
    
    // Provide a friendly string version of this instance.
    var description: String
    {
        return "(PPMLanguageModel)"
    }
}
