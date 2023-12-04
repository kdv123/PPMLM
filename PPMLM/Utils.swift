// Home for static helper functions.
// Also extensions to built-in types.

import Foundation

struct Utils
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
    
    // Calculate the perplexity given a sum of log probs (base 10).
    static func perplexity(sumLog10Prob: Double, numEvents: Int) -> Double
    {
        return pow(10, -1.0 * sumLog10Prob / Double(numEvents))
    }

    // Return memory in megabytes.
    // https://stackoverflow.com/questions/40991912/how-to-get-memory-usage-of-my-application-and-system-in-swift-by-programatically
    static func memoryInMB() -> Int
    {
        // The `TASK_VM_INFO_COUNT` and `TASK_VM_INFO_REV1_COUNT` macros are too
        // complex for the Swift C importer, so we have to define them ourselves.
        let TASK_VM_INFO_COUNT = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        guard let offset = MemoryLayout.offset(of: \task_vm_info_data_t.min_address) else {return 0}
        let TASK_VM_INFO_REV1_COUNT = mach_msg_type_number_t(offset / MemoryLayout<integer_t>.size)
        var info = task_vm_info_data_t()
        var count = TASK_VM_INFO_COUNT
        let kr = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), intPtr, &count)
            }
        }
        guard
            kr == KERN_SUCCESS,
            count >= TASK_VM_INFO_REV1_COUNT
        else { return 0 }
        
        let usedBytes = Float(info.phys_footprint)
        let usedBytesInt: UInt64 = UInt64(usedBytes)
        return Int(usedBytesInt / 1024 / 1024)
    }
    
    // Read all the lines from a given filename.
    // Returns a list of all the lines.
    static func readLinesFrom(filename: String) throws -> [String]
    {
        var lines = [String]()
        let fileURL = URL(fileURLWithPath: filename)
        let trainData = try Data(contentsOf: fileURL)
        if let trainLines = String(data: trainData, encoding: .utf8)
        {
            lines = trainLines.lines
        }
        return lines
    }
    
    
    // Sum the length of an array of strings.
    static func countCharacters(texts: [String]) -> Int
    {
        var chars = 0
        for line in texts
        {
            chars += line.count
        }
        return chars
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
