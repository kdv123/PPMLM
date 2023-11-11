// Represents the current context within the PPM suffix tree.
// An object of this type needs to be passed in to make predictions.

class Context: CustomStringConvertible
{
    var order: Int
    var head: Node

    init(order: Int, head: Node)
    {
        self.order = order
        self.head = head
    }
    
    var description: String 
    {
        return "Context [order \(order) head \(head)]"
    }
}
