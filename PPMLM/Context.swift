// Represents the current context within the PPM suffix tree.
// An object of this type needs to be passed in to make predictions.

class Context: CustomStringConvertible
{
    var order: Int
    var head: Int

    init(order: Int, head: Int)
    {
        self.order = order
        self.head = head
    }
    
    // Provide a friendly string version of this instance.
    var description: String
    {
        return "Context (order \(order) head \(head)))"
    }
}
