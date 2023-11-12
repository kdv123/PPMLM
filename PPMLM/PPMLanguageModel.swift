// Prediction by partial matching (PPM) language model.
// Based on Google's JavaScript implementation: https://github.com/google-research/google-research/tree/master/jslm

class PPMLanguageModel: CustomStringConvertible
{
    private let vocab: Vocabulary
    private let root: Node
    private let rootContext: Context
    private(set) var maxOrder: Int
    private(set) var numNodes: Int
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
            
            let symbolNode = Node(symbol: symbol, next: toNode.child, backoff: backoff)
            toNode.child = symbolNode
            numNodes += 1
            return symbolNode
        }
    }
        
    // Creates new context which is initially empty.
    func createContext() -> Context
    {
        return Context(order: rootContext.order, head: rootContext.head)
    }
    
    // Clones existing context.
    func clone(context: Context) -> Context
    {
        return Context(order: context.order, head: context.head)
    }

    // Adds symbol to the supplied context. Does not update the model.
    func addSymbolToContext(context: Context, symbol: Int)
    {
        // Only add valid symbols.
        if symbol >= vocab.size || symbol < Constants.ROOT_SYMBOL
        {
            // Crash in debug, just return in release.
            assert(symbol < vocab.size && symbol >= Constants.ROOT_SYMBOL, "Invalid symbol: \(symbol)")
            return
        }

        // Loop will be modifying the non-optional head of context.
        // So we'll just loop until we either extend an existing context
        // or give up and go all the way back to the root.
        while true
        {
            if context.order < maxOrder
            {
                // Extend the current context.
                let childNode = context.head.findChildWith(symbol: symbol)
                if let childNodeUnwrapped = childNode
                {
                    context.head = childNodeUnwrapped
                    context.order += 1
                    return
                }
            }
            // Try to extend the shorter context.
            context.order -= 1
            if let backoff = context.head.backoff
            {
                context.head = backoff
            }
            else
            {
                context.head = root
                context.order = 0
                return
            }
        }
    }

    // Adds symbol to the supplied context and update the model.
    func addSymbolAndUpdate(context: Context, symbol: Int)
    {
        // Only add valid symbols.
        if symbol >= vocab.size || symbol < Constants.ROOT_SYMBOL
        {
            // Crash in debug, just return in release.
            assert(symbol < vocab.size && symbol >= Constants.ROOT_SYMBOL, "Invalid symbol: \(symbol)")
            return
        }
                
        let symbolNode = add(symbol: symbol, toNode: context.head)
        assert(symbolNode === context.head.findChildWith(symbol: symbol), "failed to find added child")

        context.head = symbolNode
        context.order += 1
        
        // TODO: Do we really need a loop here? Can't we only go over by 1?
        while context.order > maxOrder
        {
            if let backoff = context.head.backoff
            {
                context.head = backoff
                context.order -= 1
            }
            else
            {
                assertionFailure("backoff was nil while shortening after add")
            }
        }
    }
 
    // Returns probabilities for all the symbols in the vocabulary given the
    // context.
    //
    // Notation:
    // ---------
    //         $x_h$ : Context representing history, $x_{h-1}$ shorter context.
    //   $n(w, x_h)$ : Count of symbol $w$ in context $x_h$.
    //      $T(x_h)$ : Total count in context $x_h$.
    //      $q(x_h)$ : Number of symbols with non-zero counts seen in context
    //                 $x_h$, i.e. |{w' : c(x_h, w') > 0}|. Alternatively, this
    //                 represents the number of distinct extensions of history
    //                 $x_h$ in the training data.
    //
    // Standard Kneser-Ney method (aka Absolute Discounting):
    // ------------------------------------------------------
    // Subtracting \beta (in [0, 1)) from all counts.
    //   P_{kn}(w | x_h) = \frac{\max(n(w, x_h) - \beta, 0)}{T(x_h)} +
    //                     \beta * \frac{q(x_h)}{T(x_h)} * P_{kn}(w | x_{h-1}),
    // where the second term in summation represents escaping to lower-order
    // context.
    //
    // See: Ney, Reinhard and Kneser, Hermann (1995): “Improved backing-off for
    // M-gram language modeling”, Proc. of Acoustics, Speech, and Signal
    // Processing (ICASSP), May, pp. 181–184.
    //
    // Modified Kneser-Ney method (Dasher version [3]):
    // ------------------------------------------------
    // Introducing \alpha parameter (in [0, 1)) and estimating as
    //   P_{kn}(w | x_h) = \frac{\max(n(w, x_h) - \beta, 0)}{T(x_h) + \alpha} +
    //                     \frac{\alpha + \beta * q(x_h)}{T(x_h) + \alpha} *
    //                     P_{kn}(w | x_{h-1}) .
    //
    // Additional details on the above version are provided in Sections 3 and 4
    // of:
    //   Steinruecken, Christian and Ghahramani, Zoubin and MacKay, David (2016):
    //   "Improving PPM with dynamic parameter updates", In Proc. Data
    //   Compression Conference (DCC-2015), pp. 193--202, April, Snowbird, UT,
    //   USA. IEEE.
    func getProbs(context: Context) -> [Double]
    {
        // Initialize the initial estimates. Note, we don't use uniform
        // distribution here.
        var probs = [Double](repeating: 0.0, count: vocab.size)
        
        // Initialize the exclusion mask.
        // We'll add symbol IDs to the set if they are to be excluded.
        var exclusionMask = Set<Int>()
        
        // Estimate the probabilities for all the symbols in the supplied context.
        // This runs over all the symbols in the context and over all the suffixes
        // (orders) of the context. If the exclusion mechanism is enabled, the
        // estimate for a higher-order ngram is fully trusted and is excluded from
        // further interpolation with lower-order estimates.
        //
        // Exclusion mechanism is disabled by default. Enable it with care: it has
        // been shown to work well on large corpora, but may in theory degrade the
        // performance on smaller sets (as we observed with default Dasher English
        // training data).
        var totalMass = 1.0;
        var gamma = totalMass

        // Since Context's head is non-optional, but we want to loop until this
        // helper variable node becomes nil, we need to make it into an optional.
        var node: Node? = context.head

        while let nodeUnwrapped = node
        {
            let count = nodeUnwrapped.totalChildrenCounts(exclusionMask: exclusionMask)
            if count > 0
            {
                var childNode = nodeUnwrapped.child
                while let childNodeUnwrapped = childNode
                {
                    let symbol = childNodeUnwrapped.symbol
                    if !useExclusion || !exclusionMask.contains(symbol)
                    {
                        let p = gamma * (Double(childNodeUnwrapped.count) - Constants.KN_BETA) / (Double(count) + Constants.KN_ALPHA)
                        probs[symbol] += p
                        totalMass -= p
                        if useExclusion
                        {
                            exclusionMask.insert(symbol)
                        }
                    }
                    childNode = childNodeUnwrapped.next
                }
            }
            
            // Backoff to lower-order context. The $\gamma$ factor represents the
            // total probability mass after processing the current $n$-th order before
            // backing off to $n-1$-th order. It roughly corresponds to the usual
            // interpolation parameter, as used in the literature, e.g. in
            //   Stanley F. Chen and Joshua Goodman (1999): "An empirical study of
            //   smoothing techniques for language modeling", Computer Speech and
            //   Language, vol. 13, pp. 359-–394.
            //
            // Note on computing $gamma$:
            // --------------------------
            // According to the PPM papers, and in particular the Section 4 of
            //   Steinruecken, Christian and Ghahramani, Zoubin and MacKay,
            //   David (2016): "Improving PPM with dynamic parameter updates", In
            //   Proc. Data Compression Conference (DCC-2015), pp. 193--202, April,
            //   Snowbird, UT, USA. IEEE,
            // that describes blending (i.e. interpolation), the second multiplying
            // factor in the interpolation $\lambda$ for a given suffix node $x_h$ in
            // the tree is given by
            //   \lambda(x_h) = \frac{q(x_h) * \beta + \alpha}{T(x_h) + \alpha} .
            // It can be shown that
            //   \gamma(x_h) = 1.0 - \sum_{w'}
            //      \frac{\max(n(w', x_h) - \beta, 0)}{T(x_h) + \alpha} =
            //      \lambda(x_h)
            // and, in the update below, the following is equivalent:
            //   \gamma = \gamma * \gamma(x_h) = totalMass .
            //
            // Since gamma *= (numChildren * knBeta + knAlpha) / (count + knAlpha) is
            // expensive, we assign the equivalent totalMass value to gamma.
            node = nodeUnwrapped.backoff
            gamma = totalMass
        }
        assert(totalMass >= 0.0, "Invalid remaining probability mass: \(totalMass)")
        
        // Count the total number of symbols that should have their estimates
        // blended with the uniform distribution representing the zero context.
        // When exclusion mechanism is enabled (by enabling this.useExclusion_)
        // this number will represent the number of symbols not seen during the
        // training, otherwise, this number will be equal to total number of
        // symbols because we always interpolate with the estimates for an empty
        // context.
        let numUnseenSymbols = useExclusion ? exclusionMask.count : vocab.size - 1
                
        // Adjust the probability mass for all the symbols.
        let p = totalMass / Double(numUnseenSymbols)
        for i in 1..<vocab.size
        {
            // Following is estimated according to a uniform distribution
            // corresponding to the context length of zero.
            if !useExclusion || !exclusionMask.contains(i)
            {
                
                probs[i] += p
                totalMass -= p
            }
        }
        
        var leftSymbols = vocab.size - 1
        var newProbMass = 0.0;
        for i in 1..<vocab.size
        {
            let p = totalMass / Double(leftSymbols)
            probs[i] += p;
            totalMass -= p;
            newProbMass += probs[i];
            leftSymbols -= 1
        }
        
        assert(totalMass == 0.0, "Expected remaining probability mass to be zero!")
        assert(abs(1.0 - newProbMass) < Constants.EPSILON, "Leftover mass is too big!")
        
        // All the cool kids sum to 1
        assert(abs(1.0 - probs.sum()) < Constants.EPSILON, "Probability distribution does not sum to 1.0!")
        
        return probs
    }

    // Convience function that provides the probability distribution as a dictionary
    // mapping token strings to the probability. Drops the root context.
    func getProbsAsDictionary(context: Context) -> [String: Double]
    {
        let probs = getProbs(context: context)
        var result = [String: Double]()
        
        // Loop over all the probabilities and create the dictionary entry.
        for i in 1..<probs.count
        {
            let token = vocab.getToken(ofSymbol: i)
            if let token = token
            {
                result[token] = probs[i]
            }
        }
        return result
    }
    
    // Helper function for printing out the suffix tree.
    // Returns count of printed nodes (sanity check).
    private func printTree(node: Node, indent: String) -> Int
    {
        var result = 1
        print("\(indent)\(node)")
        let indentMore = indent + "  "
        
        var current = node.child
        
        while let currentUnwrapped = current
        {
            result += printTree(node: currentUnwrapped, indent: indentMore)
            current = currentUnwrapped.next
        }
        return result
    }

    // Print out the suffix tree showing all the nodes.
    func printTree()
    {
        let count = printTree(node: root, indent: "")
        assert(count == numNodes, "Printed number of nodes does not match class counter!")
        print("Total nodes including root: \(count)")
    }
    
    // Provide a friendly string version of this instance.
    var description: String
    {
        return "(PPMLanguageModel numNodes \(numNodes) maxOrder \(maxOrder) useExclusion \(useExclusion))"
    }
    
    // Use the given line of text to update counts in the language model.
    // This assumes:
    //   1) Training starts at the root context.
    //   2) Tokens not in the vocabulary are skipped over.
    //
    // Returns number of skipped tokens.
    func train(text: String) -> Int
    {
        var skipped = 0
                    
        // Start at the root context.
        let c = createContext()
        
        // Loop over each character in the string.
        for ch in text
        {
            let symbol = vocab.getSymbol(ofToken: String(ch))
            if let symbol = symbol
            {
                addSymbolAndUpdate(context: c, symbol: symbol)
            }
            else
            {
                skipped += 1
            }
        }
        
        return skipped
    }
}
