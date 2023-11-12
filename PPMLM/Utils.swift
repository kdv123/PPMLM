// Home for static helper functions

class Utils
{
    // Determine if a reference to an object is not nil
    static func isObjectNotNil(object:AnyObject!) -> Bool
    {
        if let _:AnyObject = object
        {
            return true
        }
        return false
    }
    
    // Determine if a reference to an object is nil.
    static func isObjectNil(object:AnyObject!) -> Bool
    {
        if let _:AnyObject = object
        {
            return false
        }
        return true
    }
    
}
