// Prediction by partial matching (PPM) language model.
// Based on Google's JavaScript implementation: https://github.com/google-research/google-research/tree/master/jslm

import Foundation

struct PPMLanguageModel: CustomStringConvertible
{
    private let vocab: Vocabulary
    private let rootContext: Context
    private(set) var maxOrder: Int
    private(set) var numNodes: Int
    var useExclusion: Bool = false
    
    // Rather than store pointers in the Nodes, we'll store index into this array.
    private var listNodes = [Node]()
    
    init(vocab: Vocabulary, maxOrder: Int, reserveCapacity: Int? = nil)
    {
        self.vocab = vocab
        assert(vocab.size > 1, "Expecting at least two symbols in the vocabulary")
        
        if let reserveCapacity = reserveCapacity
        {
            listNodes.reserveCapacity(reserveCapacity)
        }
        
        self.maxOrder = maxOrder
        // First element in the array will be the root node.
        listNodes.append(Node())
        self.rootContext = Context(order: 0, head: 0)
        self.numNodes = 1
    }
    
    // Add the specified symbol to the Node specified by its array index.
    // Returns index of the existing or the newly created Node.
    private mutating func add(symbol: Int, toNodeIndex: Int) -> Int
    {
        let symbolIndex = findChildWith(nodeIndex: toNodeIndex, symbol: symbol)
        
        if symbolIndex != Node.NULL
        {
            // Update the counts for the given node.  Only updates the counts for
            // the highest order already existing node for the symbol ('single
            // counting' or 'update exclusion').
            listNodes[symbolIndex].incrementCount()
            return symbolIndex
        }
        else
        {
            // Symbol does not exist under the given node. Create a new child node
            // and update the backoff structure for lower contexts.

            // Figure out the backoff Node to use for creating the new Node.
            var backoffIndex: Int = 0  // Shortest possible context.
            if toNodeIndex != 0
            {
                let toNodeBackoffIndex = listNodes[toNodeIndex].backoff
                backoffIndex = add(symbol: symbol, toNodeIndex: toNodeBackoffIndex)
            }
            
            // Add a new node to our list.
            let symbolIndex = listNodes.count
            listNodes.append(Node(symbol: symbol, next: listNodes[toNodeIndex].child, backoff: backoffIndex))
            listNodes[toNodeIndex].child = symbolIndex
            numNodes += 1
            
            return symbolIndex
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
                let childNodeIndex = findChildWith(nodeIndex: context.head, symbol: symbol)
                if childNodeIndex != Node.NULL
                {
                    context.head = childNodeIndex
                    context.order += 1
                    return
                }
            }
            // Try to extend the shorter context.
            context.order -= 1
            let backoff = listNodes[context.head].backoff
            if backoff != Node.NULL
            {
                context.head = backoff
            }
            else
            {
                context.head = 0
                context.order = 0
                return
            }
        }
    }

    // Adds symbol to the supplied context and update the model.
    mutating func addSymbolToContextAndUpdate(context: Context, symbol: Int)
    {
        // Only add valid symbols.
        if symbol >= vocab.size || symbol < Constants.ROOT_SYMBOL
        {
            // Crash in debug, just return in release.
            assert(symbol < vocab.size && symbol >= Constants.ROOT_SYMBOL, "Invalid symbol: \(symbol)")
            return
        }
                
        let symbolNode = add(symbol: symbol, toNodeIndex: context.head)
        assert(symbolNode == findChildWith(nodeIndex: context.head, symbol: symbol), "failed to find added child")

        context.head = symbolNode
        context.order += 1
        
        // TODO: Do we really need a loop here? Can't we only go over by 1?
        while context.order > maxOrder
        {
            let backoff = listNodes[context.head].backoff
            if backoff != Node.NULL
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
        var nodeIndex = context.head

        while nodeIndex != Node.NULL
        {
            let count = totalChildrenCounts(nodeIndex: nodeIndex, exclusionMask: exclusionMask)
            if count > 0
            {
                var childNodeIndex = listNodes[nodeIndex].child
                while childNodeIndex != Node.NULL
                {
                    let symbol = listNodes[childNodeIndex].symbol
                    if !useExclusion || !exclusionMask.contains(symbol)
                    {
                        let p = gamma * (Double(listNodes[childNodeIndex].count) - Constants.KN_BETA) / (Double(count) + Constants.KN_ALPHA)
                        probs[symbol] += p
                        totalMass -= p
                        if useExclusion
                        {
                            exclusionMask.insert(symbol)
                        }
                    }
                    childNodeIndex = listNodes[childNodeIndex].next
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
            nodeIndex = listNodes[nodeIndex].backoff
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
    private func printTree(nodeIndex: Int, indent: String) -> Int
    {
        var result = 1
        print("\(indent)\(nodeIndex)")
        let indentMore = indent + "  "
        
        var current = listNodes[nodeIndex].child
        
        while current != Node.NULL
        {
            result += printTree(nodeIndex: current, indent: indentMore)
            current = listNodes[current].next
        }
        return result
    }

    // Print out the suffix tree showing all the nodes.
    func printTree()
    {
        let count = printTree(nodeIndex: 0, indent: "")
        assert(count == numNodes, "Printed number of nodes does not match class counter!")
        print("Total nodes including root: \(count)")
    }

    // Helper that descends tree summing stats.
    private func statsTree(nodeIndex: Int) -> TreeStats
    {
        var currentIndex = listNodes[nodeIndex].child
        var result = TreeStats()
        var childCount = 0
        
        while currentIndex != Node.NULL
        {
            let childResult = statsTree(nodeIndex: currentIndex)
            result.nodes += childResult.nodes + 1
            result.leaves += childResult.leaves
            result.singletons += childResult.singletons
            let count = listNodes[currentIndex].count
            if count == 1
            {
                result.singletons += 1
            }
            if count > result.maxCount
            {
                result.maxCount = count
            }
            currentIndex = listNodes[currentIndex].next
            childCount += 1
        }
        if childCount == 0
        {
            result.leaves += 1
        }
        
        return result
    }

    // Calculates various statistics about the tree.
    func statsTree() -> TreeStats
    {
        var result = statsTree(nodeIndex: 0)
        result.nodes += 1
        return result
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
    //   3) Each character in the string is considered a token.
    // Returns number of skipped tokens.
    mutating func train(text: String) -> Int
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
                addSymbolToContextAndUpdate(context: c, symbol: symbol)
            }
            else
            {
                skipped += 1
            }
        }
        
        return skipped
    }
    
    // Trains on a set of strings.
    // Each element is trained on separately and assumed conditions on root context.
    // Returns number of skipped tokens.
    mutating func train(texts: [String]) -> Int
    {
        var skipped = 0
        for text in texts
        {
            skipped += train(text: text)
        }
        return skipped
    }
    
    // Compute the log (base 10) probability of some text.
    // This assume we condition the first character on the root symbol.
    // Be default, the model's counts are not updated during evaluation.
    // Skips over any characters not in the vocab.
    // There is no end of sentence symbol, so number of events is length of text.
    // Returns a tuple containing various stats about the evaluation.
    mutating func evaluate(text: String, updateModel: Bool = false) ->
        (sumLogProb: Double, tokensGood: Int, tokensSkipped: Int, perplexity: Double)
    {
        let c = createContext()
        var sumLogProb = 0.0
        var tokensGood = 0
        var tokensSkipped = 0
        
        for ch in text
        {
            // Try and map this character to a numeric symbol ID
            let nextSymbol = vocab.getSymbol(ofToken: String(ch))
            if let nextSymbol = nextSymbol
            {
                sumLogProb += log10(getProbs(context: c)[nextSymbol])
                tokensGood += 1
                if updateModel
                {
                    addSymbolToContextAndUpdate(context: c, symbol: nextSymbol)
                }
                else
                {
                    addSymbolToContext(context: c, symbol: nextSymbol)
                }
            }
            else
            {
                tokensSkipped += 1
            }
        }
        return (sumLogProb, tokensGood, tokensSkipped, Utils.perplexity(sumLog10Prob: sumLogProb, numEvents: tokensGood))
    }
    
    // Convience function that evaluates on a set of texts.
    mutating func evaluate(texts: [String], updateModel: Bool = false) ->
        (sumLogProb: Double, tokensGood: Int, tokensSkipped: Int, perplexity: Double)
    {
        var sumLogProb = 0.0
        var tokensGood = 0
        var tokensSkipped = 0
        
        for text in texts
        {
            let result = evaluate(text: text, updateModel: updateModel)
            sumLogProb += result.sumLogProb
            tokensGood += result.tokensGood
            tokensSkipped += result.tokensSkipped
        }
        return (sumLogProb, tokensGood, tokensSkipped, Utils.perplexity(sumLog10Prob: sumLogProb, numEvents: tokensGood))
    }
    
    // Finds child of the specified node with a specified symbol.
    // Returns the index of the Node object that matches the symbol.
    // If no match, returns Node.NULL.
    private func findChildWith(nodeIndex: Int, symbol: Int) -> Int
    {
        var currentIndex = listNodes[nodeIndex].child
        // Loop until we hit the end of the linked list.
        while currentIndex != Node.NULL
        {
            if (listNodes[currentIndex].symbol == symbol)
            {
                // Found the desired symbol.
                return currentIndex;
            }
            currentIndex = listNodes[currentIndex].next
        }
        return currentIndex;
    }
    
    // Total number of observations for all the children of this node. This
    // counts all the events observed in this context.
    //
    // Note: This API is used at inference time. A possible alternative that will
    // speed up the inference is to store the number of children in each node as
    // originally proposed by Moffat for PPMB in
    //   Moffat, Alistair (1990): "Implementing the PPM data compression scheme",
    //   IEEE Transactions on Communications, vol. 38, no. 11, pp. 1917--1921.
    // This however will increase the memory use of the algorithm which is already
    // quite substantial.
    private func totalChildrenCounts(nodeIndex: Int, exclusionMask: Set<Int>?) -> Int
    {
        var currentIndex = listNodes[nodeIndex].child
        var count = 0
        // Loop until we hit the end of the linked list.
        while currentIndex != Node.NULL
        {
            if let exclusionMaskUnwrapped = exclusionMask
            {
                if !exclusionMaskUnwrapped.contains(listNodes[currentIndex].symbol)
                {
                    count += listNodes[currentIndex].count
                }
            }
            else
            {
                // No exclusion mask specified, sum all the children
                count += listNodes[currentIndex].count
            }
            currentIndex = listNodes[currentIndex].next
        }
        return count;
    }

}
