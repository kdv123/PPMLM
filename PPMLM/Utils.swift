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
    
}

extension Sequence where Element: Numeric 
{
    // Returns the sum of all elements in the collection
    func sum() -> Element { return reduce(0, +) }
}
