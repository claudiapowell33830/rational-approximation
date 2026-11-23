# rational-approximation

Fits rational functions to arbitrary univariate functions using Padé and minimax (Remez) approximation methods.

## Usage

```go
package main

import (
	"fmt"
	"math"

	rational "github.com/example/rational-approximation"
)

func main() {
	// Padé approximant of exp(x) of order [2/2].
	// Taylor coefficients up to order 4.
	coeffs := []float64{1, 1, 1.0 / 2, 1.0 / 6, 1.0 / 24}
	r, err := rational.Pade(coeffs, 2, 2)
	if err != nil {
		panic(err)
	}
	fmt.Println(r.Eval(1)) // ~2.7142857142857144

	// Minimax fit of sin(x) on [0, pi/2] with a [1/1] rational function.
	f := func(x float64) float64 { return math.Sin(x) }
	r2, maxErr, iter, err := rational.Minimax(f, 0, math.Pi/2, 1, 1, nil, 100)
	if err != nil {
		panic(err)
	}
	fmt.Println(r2.Eval(1), maxErr, iter)
}
```

## Why this library exists

Approximating a function by a polynomial can require many terms to achieve a given accuracy, especially for functions with poles or rapid variation. Rational functions often achieve the same accuracy with much lower degree. This library provides two standard ways to build them: Padé approximation, which matches Taylor coefficients, and minimax approximation, which minimizes the maximum error on an interval. The trade-off is that minimax fitting is iterative and may not converge for arbitrary functions; the caller must check the returned error and iteration count.

## Edge cases

- The Padé method requires enough Taylor coefficients and a non-singular linear system. If the system is singular, it returns an error.
- The minimax method uses a fixed dense grid (1000 points) to estimate the maximum error. This is a deliberate simplification: it is deterministic and fast, but can miss narrow spikes in the error function. For well-behaved functions on modest intervals it works well.
- Evaluation at a pole returns Inf or NaN; the caller should avoid points where the denominator is zero.
