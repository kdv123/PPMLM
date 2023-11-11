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
    // Node containing the linked list that has symbols extending this node by one more symbol.
    private var child: Node?
    
    // Next node in the linked list for seen symbols after our current Node's context.
    private var next: Node?

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
    private var backoff: Node?
    
    // Frequency count for this node. Number of times the suffix symbol stored
    // in this node was observed.
    private var count: Int = 0

    // Symbol that this node stores.
    private var symbol: Int = Constants.ROOT_ID
    
    // Finds child of the current node with a specified symbol.
    func findChildWith(symbol: Int) -> Node?
    {
        var current = child
        // Loop until we hit the end of the linked list.
        while let currentUnwrapped = current
        {
            if (currentUnwrapped.symbol == symbol)
            {
                // Found the desired symbol.
                return currentUnwrapped;
            }
            current = currentUnwrapped.next
        }
        return current;
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
    func totalChildrenCounts(exclusionMask: Set<Int>?) -> Int
    {
        var current = child
        var count = 0
        // Loop until we hit the end of the linked list.
        while let currentUnwrapped = current
        {
            if let exclusionMaskUnwrapped = exclusionMask
            {
                if !exclusionMaskUnwrapped.contains(currentUnwrapped.symbol)
                {
                    count += currentUnwrapped.count
                }
            }
            else
            {
                // No exclusion mask specified, sum all the children
                count += currentUnwrapped.count
            }
            current = currentUnwrapped.next
        }
        return count;
    }
    
    // Provide a friendly string version of this instance.
    var description: String
    {
        return "(Node symbol \(symbol) count \(count))"
    }
}
