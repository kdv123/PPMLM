// main program that tests out the PPMLM class

import Foundation

//print(malloc_size(Unmanaged.passRetained(c).toOpaque()))
//print(class_getInstanceSize(type(of: c)))

var v = Vocabulary()
let aSymbol = v.add(symbol: "a")
let bSymbol = v.add(symbol: "b")

let maxOrder = 5
var lm = PPMLanguageModel(vocab: v, maxOrder: maxOrder)
var c = lm.createContext()
lm.addSymbolAndUpdate(context: c, symbol: aSymbol)
lm.addSymbolAndUpdate(context: c, symbol: bSymbol)
print("Initial count trie:")
lm.printTree()

c = lm.createContext();
var probs = lm.getProbs(context: c)
print(probs)
assert(probs.count == 3, "Expected \"a\", \"b\" and root")

// Nothing has been entered yet. Since we've observed both "a" and "b", there is
// an equal likelihood of getting either.
assert(probs[1] > 0 && probs[1] == probs[2], "Probabilities for both symbols should be equal")

// Enter "a" and check the probability estimates. Since we've seen the sequence
// "ab" during the training, the "b" should be more likely than "a".
lm.addSymbolToContext(context: c, symbol: aSymbol)
probs = lm.getProbs(context: c)
print(probs)
assert(probs[1] > 0 && probs[1] < probs[2], "Probability for \"b\" should be more likely")

// Enter "b". The context becomes "ab". Now it's back to square one: Any symbol
// is likely again.
lm.addSymbolToContext(context: c, symbol: bSymbol)
probs = lm.getProbs(context: c)
print(probs)
assert(probs[1] > 0 && probs[1] == probs[2], "Probabilities for both symbols should be equal")
