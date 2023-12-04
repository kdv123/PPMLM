// Prediction by partial matching (PPM) language model.
// Based on Google's JavaScript implementation: https://github.com/google-research/google-research/tree/master/jslm
//
// Compared to the JavaScript implementation, this has been refactored to reduce memory by:
//  1) Storing suffix tree in parallel arrays rather than a linked structure.
//  2) Reducing to a vocabulary size of 256 symbols.
//  3) Reducing maximum count of any node to 2^32.

import Foundation

struct PPMLanguageModel: CustomStringConvertible
{
    // Types we use to store various things in the suffix tree.
    // These can be changed if a langauge or training set exceeds the limit.
    typealias Symbol = UInt8            // Type used to encode a character, data type determines max vocab size
    typealias Count = UInt32            // Type for the count for a given node, data type determines max count
    typealias NodeIndex = UInt32        // Type used in parallel arrays, data type determines the max allowed nodes
        
    static let NULL = NodeIndex.max     // Constant that denotes the end of a linked list
    
    private let vocab: Vocabulary       // Handles mapping characters to integer symbol IDs
    private let rootContext: Context
    private(set) var maxOrder: Int      // Maximum order of the model
    private(set) var numNodes: Int      // Count of the number of nodes created
    private var capacityIncrease = 0    // How many extra array entries to reserve if parallel arrays are full, 0=default doubling
    
    var useExclusion: Bool = false
        
    // Store the symbols, counts in a parallel array.
    // This saves memory since padding in struct wastes bytes for types less than 4 bytes.
    // Also made training and evaluating faster, probably by saving function calls and casts?
    // The integer index of elements serve as the virtual pointer address of each Node.
    
    // Symbols for each node.
    private var symbols = [Symbol]()
    
    // Frequency count for this node. Number of times the suffix symbol stored was observed.
    private var counts = [Count]()
    
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
    private var backoffs = [NodeIndex]()
    
    // Next node in the linked list for seen symbols after our current Node's context.
    private var nexts = [NodeIndex]()
    
    // Node containing the linked list that has symbols extending this node by one more symbol.
    private var childs = [NodeIndex]()
    
    // Shrink our internal array capacity to the actual size (plus an optional extra amount).
    // This doesn't seem to actually reduce reported memory?
    mutating func shrink(extra: Int? = 0)
    {
        print("Shrink, old size \(symbols.count), old capacity \(symbols.capacity)")
        var newCapacity = symbols.count
        if let extra = extra
        {
            newCapacity += extra
        }
        print("Shrink, new capacity \(newCapacity)")
        
        var newSymbols = [Symbol]()
        print("Shrink, newSymbols size \(newSymbols.count) capacity1 \(newSymbols.capacity)")
        newSymbols.reserveCapacity(newCapacity)
        print("Shrink, newSymbols size \(newSymbols.count) capacity2 \(newSymbols.capacity)")
        newSymbols.append(contentsOf: symbols)
        print("Shrink, newSymbols size \(newSymbols.count) capacity3 \(newSymbols.capacity)")
        symbols.removeAll(keepingCapacity: false)
        symbols.append(contentsOf: newSymbols)
        print("Shrink, symbols size \(symbols.count) capacity3 \(symbols.capacity)")

        var newCounts = [Count]()
        newCounts.reserveCapacity(newCapacity)
        newCounts.append(contentsOf: counts)
        //counts.removeAll(keepingCapacity: false)
        counts = newCounts

        var newBackoffs = [NodeIndex]()
        newBackoffs.reserveCapacity(newCapacity)
        newBackoffs.append(contentsOf: backoffs)
        //backoffs.removeAll(keepingCapacity: false)
        backoffs = newBackoffs
        
        var newNexts = [NodeIndex]()
        newNexts.reserveCapacity(newCapacity)
        newNexts.append(contentsOf: nexts)
        //nexts.removeAll(keepingCapacity: false)
        nexts = newNexts
        
        var newChilds = [NodeIndex]()
        newChilds.reserveCapacity(newCapacity)
        newChilds.append(contentsOf: childs)
        //childs.removeAll(keepingCapacity: false)
        childs = newChilds
        
        print("Shrink, new size \(symbols.count), new capacity \(symbols.capacity) \(newSymbols.capacity)")

    }
    
    // Helper that increases the capacity of our parallel arrays.
    private mutating func extendArrays(reserveCapacity : Int)
    {
        symbols.reserveCapacity(reserveCapacity)
        counts.reserveCapacity(reserveCapacity)
        backoffs.reserveCapacity(reserveCapacity)
        nexts.reserveCapacity(reserveCapacity)
        childs.reserveCapacity(reserveCapacity)
    }
    
    // Build a language model with the given vocabulary and max order.
    // If the caller knows the number of nodes, this can be specified to avoid
    // the expense of dynamically expanding the node array and save memory.
    init(vocab: Vocabulary, maxOrder: Int, reserveCapacity: Int? = nil, capacityIncrease: Int? = nil)
    {
        self.vocab = vocab
        assert(vocab.size > 1, "Expecting at least two symbols in the vocabulary")

        self.maxOrder = maxOrder
        self.rootContext = Context(order: 0, head: 0)
        self.numNodes = 1
                
        if let reserveCapacity = reserveCapacity
        {
            //nodes.reserveCapacity(reserveCapacity)
            extendArrays(reserveCapacity: reserveCapacity)
        }
                
        // First element in the array will be the root node.
        //nodes.append(Node())
        symbols.append(Symbol(0))
        counts.append(Count(1))
        backoffs.append(PPMLanguageModel.NULL)
        nexts.append(PPMLanguageModel.NULL)
        childs.append(PPMLanguageModel.NULL)
        
       
        
        if let capacityIncrease = capacityIncrease
        {
            self.capacityIncrease = capacityIncrease
        }
    }
    
    // Add the specified symbol to the Node specified by its array index.
    // Returns index of the existing or the newly created Node.
    private mutating func add(symbol: Symbol, toNodeIndex: Int) -> Int
    {
        let symbolIndex = findChildWith(nodeIndex: toNodeIndex, symbol: symbol)
        
        if symbolIndex != PPMLanguageModel.NULL
        {
            // Update the counts for the given node.  Only updates the counts for
            // the highest order already existing node for the symbol ('single
            // counting' or 'update exclusion').
//            nodes[symbolIndex].incrementCount()
            counts[symbolIndex] += 1
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
//                let toNodeBackoffIndex = nodes[toNodeIndex].backoff
                let toNodeBackoffIndex = backoffs[toNodeIndex]
                backoffIndex = add(symbol: symbol, toNodeIndex: Int(toNodeBackoffIndex))
            }
            
            // Add a new node to our list.
//            let symbolIndex = nodes.count
            let symbolIndex = childs.count
//            listNodes.append(Node(symbol: symbol, next: listNodes[toNodeIndex].child, backoff: backoffIndex))
//            nodes.append(Node(next: nodes[toNodeIndex].child, backoff: backoffIndex))
//            nodes.append(Node(next: nodes[toNodeIndex].child))
//            nodes.append(Node())
            childs.append(PPMLanguageModel.NULL)
//            nexts.append(NodeIndex(nodes[toNodeIndex].child))
            nexts.append(NodeIndex(childs[toNodeIndex]))
            childs[toNodeIndex] = NodeIndex(symbolIndex)
//            nodes[toNodeIndex].child = symbolIndex
            symbols.append(Symbol(symbol))
            counts.append(Count(1))
            backoffs.append(NodeIndex(backoffIndex))
            
            // If optional capacity increase is enabled check if we have reaced capacity.
            if capacityIncrease > 0 && childs.count == childs.capacity
            {
                //print("Old size \(childs.count), old capacity \(childs.capacity)")
                // Increase ourselves to avoid the standard doubling behavior.
                extendArrays(reserveCapacity: childs.count + capacityIncrease)
                //print("New size \(childs.count), new capacity \(childs.capacity)")
            }
            
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
    func addSymbolToContext(context: Context, symbol: Symbol)
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
                if childNodeIndex != PPMLanguageModel.NULL
                {
                    context.head = childNodeIndex
                    context.order += 1
                    return
                }
            }
            // Try to extend the shorter context.
            context.order -= 1
//            let backoff = nodes[context.head].backoff
            let backoff = backoffs[context.head]
            if backoff != PPMLanguageModel.NULL
            {
                context.head = Int(backoff)
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
    mutating func addSymbolToContextAndUpdate(context: Context, symbol: Symbol)
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
            //let backoff = nodes[context.head].backoff
            let backoff = backoffs[context.head]
            if backoff != PPMLanguageModel.NULL
            {
                context.head = Int(backoff)
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
        var exclusionMask = Set<Symbol>()
        
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

        while nodeIndex != PPMLanguageModel.NULL
        {
            let count = Double(totalChildrenCounts(nodeIndex: nodeIndex, exclusionMask: exclusionMask))
            if count > 0
            {
                //var childNodeIndex = nodes[nodeIndex].child
                var childNodeIndex = Int(childs[nodeIndex])
                while childNodeIndex != PPMLanguageModel.NULL
                {
                    //let symbol = listNodes[childNodeIndex].symbol
                    let symbol = symbols[childNodeIndex]
                    if !useExclusion || !exclusionMask.contains(symbol)
                    {
//                        let p = gamma * (Double(nodes[childNodeIndex].count) - Constants.KN_BETA) / (Double(count) + Constants.KN_ALPHA)
                        let p = gamma * (Double(counts[childNodeIndex]) - Constants.KN_BETA) / (count + Constants.KN_ALPHA)
                        probs[Int(symbol)] += p
                        totalMass -= p
                        if useExclusion
                        {
                            exclusionMask.insert(symbol)
                        }
                    }
//                    childNodeIndex = nodes[childNodeIndex].next
                    childNodeIndex = Int(nexts[childNodeIndex])
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
            //nodeIndex = nodes[nodeIndex].backoff
            nodeIndex = Int(backoffs[nodeIndex])
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
            if !useExclusion || !exclusionMask.contains(Symbol(i))
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
            let token = vocab.getToken(ofSymbol: Symbol(i))
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
        
        //var current = nodes[nodeIndex].child
        var current = Int(childs[nodeIndex])
        
        while current != PPMLanguageModel.NULL
        {
            result += printTree(nodeIndex: current, indent: indentMore)
            //current = nodes[current].next
            current = Int(nexts[current])
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
        //var currentIndex = nodes[nodeIndex].child
        var currentIndex = Int(childs[nodeIndex])
        var result = TreeStats()
        var childCount = 0
        
        while currentIndex != PPMLanguageModel.NULL
        {
            let childResult = statsTree(nodeIndex: currentIndex)
            result.nodes += childResult.nodes + 1
            result.leaves += childResult.leaves
            result.singletons += childResult.singletons
//            let count = nodes[currentIndex].count
            let count = counts[currentIndex]
            if count == 1
            {
                result.singletons += 1
            }
            if count > result.maxCount
            {
                result.maxCount = Int(count)
            }
            //currentIndex = nodes[currentIndex].next
            currentIndex = Int(nexts[currentIndex])
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
        print("Array count \(symbols.count), capacity \(symbols.capacity)")
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
                addSymbolToContextAndUpdate(context: c, symbol: Symbol(symbol))
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
                sumLogProb += log10(getProbs(context: c)[Int(nextSymbol)])
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
    // If no match, returns PPMLanguageModel.NULL.
    private func findChildWith(nodeIndex: Int, symbol: Symbol) -> Int
    {
        //var currentIndex = nodes[nodeIndex].child
        var currentIndex = Int(childs[nodeIndex])
        // Loop until we hit the end of the linked list.
        while currentIndex != PPMLanguageModel.NULL
        {
            //if (listNodes[currentIndex].symbol == symbol)
            if (symbols[currentIndex] == symbol)
            {
                // Found the desired symbol.
                return currentIndex;
            }
            //currentIndex = nodes[currentIndex].next
            currentIndex = Int(nexts[currentIndex])
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
    private func totalChildrenCounts(nodeIndex: Int, exclusionMask: Set<Symbol>?) -> Int
    {
        //var currentIndex = nodes[nodeIndex].child
        var currentIndex = Int(childs[nodeIndex])
        var count = 0
        // Loop until we hit the end of the linked list.
        while currentIndex != PPMLanguageModel.NULL
        {
            if let exclusionMaskUnwrapped = exclusionMask
            {
                //if !exclusionMaskUnwrapped.contains(listNodes[currentIndex].symbol)
                if !exclusionMaskUnwrapped.contains(symbols[currentIndex])
                {
                    //count += nodes[currentIndex].count
                    count += Int(counts[currentIndex])
                }
            }
            else
            {
                // No exclusion mask specified, sum all the children
                //count += nodes[currentIndex].count
                count += Int(counts[currentIndex])
            }
            //currentIndex = nodes[currentIndex].next
            currentIndex = Int(nexts[currentIndex])
        }
        return count;
    }

}
