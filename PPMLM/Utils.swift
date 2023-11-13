// Home for static helper functions.
// Also extensions to built-in types.

import Foundation

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
    
    // Return memory used in bytes.
    // https://stackoverflow.com/questions/40991912/how-to-get-memory-usage-of-my-application-and-system-in-swift-by-programatically
    static func memoryUsed() -> Int
    {
        var taskInfo = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) 
        {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) 
            {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        if kerr == KERN_SUCCESS 
        {
            return Int(taskInfo.resident_size)
        }
        else 
        {
            print("Error with task_info(): " +
                (String(cString: mach_error_string(kerr), encoding: String.Encoding.ascii) ?? "unknown error"))
            return 0
        }
    }
}

extension String 
{
    // Collapse consecutive whitespace into a single space.
    func condenseWhitespace() -> String
    {
        let components = self.components(separatedBy: .whitespacesAndNewlines)
        return components.filter { !$0.isEmpty }.joined(separator: " ")
    }
    
    // Remove leading and trailing whitespace.
    func trimmingLeadingAndTrailingSpaces(using characterSet: CharacterSet = .whitespacesAndNewlines) -> String
    {
        return trimmingCharacters(in: characterSet)
    }
    
    // Compute the number of unique characters.
    func numUniqueCharacters() -> Int
    {
        var seen = Set<Character>()
        for ch in self
        {
            seen.insert(ch)
        }
        return seen.count
    }
    
    // Compute the number of unique characters.
    func charactersToCount() -> [String: Int]
    {
        var result = [String: Int]()
        for ch in self
        {
            result[String(ch)] = (result[String(ch)] ?? 0) + 1
        }
        return result
    }

    var lines: [String]
    {
        return self.components(separatedBy: "\n")
    }
}

extension Sequence where Element: Numeric 
{
    // Returns the sum of all elements in the collection
    func sum() -> Element { return reduce(0, +) }
}
