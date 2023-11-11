// main program that tests out the PPMLM class

import Foundation

var c = Context(order: 1, head: Node())
print("\(c)")
//print(malloc_size(Unmanaged.passRetained(c).toOpaque()))
//print(class_getInstanceSize(type(of: c)))

var v = Vocabulary()
print("\(v)")

let aID = v.add(symbol: "a")
print("\(v)")

let bID = v.add(symbol: "b")
print("\(v)")


print("a \(aID) b \(bID)")
