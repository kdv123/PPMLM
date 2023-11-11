// main program that tests out the PPMLM class

import Foundation

//print(malloc_size(Unmanaged.passRetained(c).toOpaque()))
//print(class_getInstanceSize(type(of: c)))

var v = Vocabulary()
let aSymbol = v.add(symbol: "a")
let bSymbol = v.add(symbol: "b")
let cSymbol = v.add(symbol: "c")

let maxOrder = 5
let lm = PPMLanguageModel(vocab: v, maxOrder: maxOrder)
let c = lm.createEmptyContext()

lm.addAndUpdate(symbol: aSymbol, toContext: c)
lm.addAndUpdate(symbol: bSymbol, toContext: c)
lm.addAndUpdate(symbol: cSymbol, toContext: c)

print("lm \(lm)")
print("c \(c)")
lm.printTree()
