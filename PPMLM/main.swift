// Program that tests out the PPMLM class.
// These tests follow those in https://github.com/google-research/google-research/blob/master/jslm/example.js

import Foundation

// Create a small vocabulary.
var v = Vocabulary()
let aSymbol = v.add(symbol: "a")
let bSymbol = v.add(symbol: "b")

// Build the PPM language model trie and update the counts.
let maxOrder = 5
var lm = PPMLanguageModel(vocab: v, maxOrder: maxOrder)
var c = lm.createContext()
lm.addSymbolAndUpdate(context: c, symbol: aSymbol)
lm.addSymbolAndUpdate(context: c, symbol: bSymbol)
print("Initial count trie:")
lm.printTree()

// Check static (non-adaptive) mode.
// In the example below we always ignore the 0th symbol. It is a special symbol
// corresponding to the root of the trie.
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

// Try to enter "ba". Since the model has only observed "ab" sequence, it is
// expecting the next most probable symbol to be "b".
c = lm.createContext()
lm.addSymbolToContext(context: c, symbol: bSymbol)
lm.addSymbolToContext(context: c, symbol: aSymbol)
probs = lm.getProbs(context: c)
print(probs)
assert(probs[1] > 0 && probs[2] > probs[1], "Probability for \"b\" should be more likely")

// Check adaptive mode in which the model is updated as symbols are entered.
// Enter "a" and update the model. At this point the frequency for "a" is
// higher, so it's more probable.
lm = PPMLanguageModel(vocab: v, maxOrder: maxOrder)
c = lm.createContext()
lm.addSymbolAndUpdate(context: c, symbol: aSymbol)
probs = lm.getProbs(context: c)
print(probs)
assert(probs[1] > 0 && probs[1] > probs[2], "Probability for \"a\" should be more likely")

// Enter "b" and update the model. At this point both symbols should become
// equally likely again.
lm.addSymbolAndUpdate(context: c, symbol: bSymbol)
probs = lm.getProbs(context: c)
print(probs)
assert(probs[1] > 0 && probs[1] == probs[2], "Probabilities for both symbols should be the same")

// Enter "b" and update the model. Current context "abb". Since we've seen
// "ab" and "abb" by now, the "b" becomes more likely.
lm.addSymbolAndUpdate(context: c, symbol: bSymbol)
probs = lm.getProbs(context: c)
print(probs)
assert(probs[1] > 0 && probs[1] < probs[2], "Probability for \"b\" should be more likely")

print("Final count trie:")
lm.printTree()

//print(malloc_size(Unmanaged.passRetained(c).toOpaque()))
//print(class_getInstanceSize(type(of: c)))
