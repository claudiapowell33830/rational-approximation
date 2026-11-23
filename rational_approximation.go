// Package rationalapproximation provides rational function fitting for
// arbitrary univariate functions using Padé and minimax (Remez) methods.
//
// A rational function is a ratio of two polynomials:
//   R(x) = P(x) / Q(x)
// where P has degree m and Q has degree n. The library fits such functions
// to a target function f(x) by either matching Taylor coefficients (Padé)
// or by minimizing the maximum absolute error on a discrete grid (minimax).
//
// The minimax implementation uses the Remez exchange algorithm on a set of
// sample points. It is iterative and may not converge for pathological
// functions; the caller should check the returned error and iteration count.
package rationalapproximation

import (
	"errors"
	"math"
)

// Rational represents a rational function P(x)/Q(x).
// The coefficients are stored in ascending order: p[0] is the constant term
// of the numerator, p[1] the linear term, etc. Similarly for q.
// q[0] must be non-zero for the function to be defined at x=0, but the
// Rational type itself does not enforce this; evaluation will return NaN
// if q evaluates to zero.
type Rational struct {
	p []float64
	q []float64
}

// NewRational creates a rational function with the given numerator and
// denominator coefficients. The slices are copied. It returns an error if
// either slice is empty or if the leading denominator coefficient is zero.
func NewRational(p, q []float64) (*Rational, error) {
	if len(p) == 0 || len(q) == 0 {
		return nil, errors.New("rational: numerator and denominator must be non-empty")
	}
	if q[len(q)-1] == 0 {
		return nil, errors.New("rational: leading denominator coefficient must be non-zero")
	}
	pc := make([]float64, len(p))
	copy(pc, p)
	qc := make([]float64, len(q))
	copy(qc, q)
	return &Rational{p: pc, q: qc}, nil
}

// Eval evaluates the rational function at x.
// If the denominator evaluates to zero, the result is +Inf, -Inf, or NaN
// depending on the sign of the numerator and the approach to the pole.
func (r *Rational) Eval(x float64) float64 {
	num := polyEval(r.p, x)
	den := polyEval(r.q, x)
	return num / den
}

// Num returns a copy of the numerator coefficients.
func (r *Rational) Num() []float64 {
	out := make([]float64, len(r.p))
	copy(out, r.p)
	return out
}

// Den returns a copy of the denominator coefficients.
func (r *Rational) Den() []float64 {
	out := make([]float64, len(r.q))
	copy(out, r.q)
	return out
}

// polyEval evaluates a polynomial with coefficients in ascending order
// using Horner's method.
func polyEval(coeffs []float64, x float64) float64 {
	if len(coeffs) == 0 {
		return 0
	}
	res := coeffs[len(coeffs)-1]
	for i := len(coeffs) - 2; i >= 0; i-- {
		res = res*x + coeffs[i]
	}
	return res
}

// Pade computes the [m/n] Padé approximant of the function f at x=0.
// The function f must provide its Taylor coefficients up to order m+n.
// The coefficients are given in ascending order: coeffs[k] is the coefficient
// of x^k in the Taylor expansion of f around 0.
//
// The Padé approximant is the unique rational function P/Q with deg(P)=m,
// deg(Q)=n, Q(0)=1, and whose Taylor expansion matches f up to order m+n.
//
// It returns an error if the linear system for the denominator coefficients
// is singular, which can happen if the Taylor coefficients do not define a
// unique Padé approximant.
func Pade(coeffs []float64, m, n int) (*Rational, error) {
	if m < 0 || n < 0 {
		return nil, errors.New("rational: degrees must be non-negative")
	}
	if len(coeffs) < m+n+1 {
		return nil, errors.New("rational: not enough Taylor coefficients")
	}

	// Solve for denominator coefficients q[1..n] using the linear system:
	// sum_{j=1..n} q[j] * c[m+i-j] = -c[m+i]  for i=1..n
	// where q[0]=1.
	// Build the n x n matrix A and right-hand side b.
	A := make([][]float64, n)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		A[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			A[i][j] = coeffs[m+i-j]
		}
		b[i] = -coeffs[m+i+1]
	}

	q, err := solveLinear(A, b)
	if err != nil {
		return nil, err
	}

	// Prepend q[0] = 1.
	qFull := make([]float64, n+1)
	qFull[0] = 1
	copy(qFull[1:], q)

	// Compute numerator coefficients p[0..m] using:
	// p[k] = sum_{j=0..min(k,n)} q[j] * c[k-j]
	p := make([]float64, m+1)
	for k := 0; k <= m; k++ {
		sum := 0.0
		for j := 0; j <= k && j <= n; j++ {
			sum += qFull[j] * coeffs[k-j]
		}
		p[k] = sum
	}

	return NewRational(p, qFull)
}

// Minimax fits a rational function of degrees (m, n) to the function f
// on the interval [a, b] by minimizing the maximum absolute error.
//
// The algorithm requires an initial set of m+n+2 sample points. If samples
// is nil, a uniform grid is used. The function f is evaluated at each sample.
//
// The implementation uses the Remez exchange algorithm. It returns the fitted
// rational function, the maximum absolute error on the final sample set, and
// the number of iterations performed. If the algorithm does not converge
// within maxIter iterations, it returns the best result found and an error.
//
// The caller must ensure that f is continuous and finite on [a, b].
func Minimax(f func(float64) float64, a, b float64, m, n int, samples []float64, maxIter int) (*Rational, float64, int, error) {
	if m < 0 || n < 0 {
		return nil, 0, 0, errors.New("rational: degrees must be non-negative")
	}
	if a >= b {
		return nil, 0, 0, errors.New("rational: interval must have a < b")
	}
	if f == nil {
		return nil, 0, 0, errors.New("rational: function must not be nil")
	}
	if maxIter <= 0 {
		maxIter = 100
	}

	numPoints := m + n + 2
	if samples == nil {
		samples = make([]float64, numPoints)
		for i := 0; i < numPoints; i++ {
			samples[i] = a + (b-a)*float64(i)/float64(numPoints-1)
		}
	} else if len(samples) < numPoints {
		return nil, 0, 0, errors.New("rational: not enough sample points")
	}

	// Initial guess: use Padé if possible, otherwise a constant.
	// We'll start with a linear system solve using the first numPoints samples.
	// The Remez algorithm iteratively updates the sample set.

	// We need to solve for the coefficients of P and Q such that
	// P(x_i) - f(x_i) * Q(x_i) = (-1)^i * E
	// where E is the error. This is a linear system of size (m+1)+(n+1)+1.
	// We fix Q(0)=1 to remove the scaling ambiguity, but here we can fix
	// one coefficient of Q to 1. We'll fix q[0]=1.

	// The system has unknowns: p[0..m], q[1..n], E.
	// Number of unknowns: (m+1) + n + 1 = m+n+2.

	// We'll use a simple Gaussian elimination solver.

	// Since the Remez algorithm can be unstable, we limit iterations.

	// We'll follow the standard approach: solve the linear system for the
	// current sample set, then find the point of maximum error on a dense grid,
	// and exchange it with one of the sample points.

	// To keep the implementation self-contained and dependency-free, we use
	// a fixed dense grid for error evaluation. This is a trade-off: it may
	// miss narrow spikes, but it is simple and deterministic.

	// Dense grid for error evaluation.
	const densePoints = 1000

	// Initialize sample set.
	xs := make([]float64, numPoints)
	copy(xs, samples[:numPoints])

	var bestR *Rational
	bestErr := math.Inf(1)

	for iter := 0; iter < maxIter; iter++ {
		// Build and solve the linear system for the current sample set.
		r, err := solveRemezSystem(f, xs, m, n)
		if err != nil {
			return nil, 0, iter, err
		}

		// Evaluate maximum error on a dense grid.
		maxErr := 0.0
		maxX := a
		for i := 0; i <= densePoints; i++ {
			x := a + (b-a)*float64(i)/float64(densePoints)
			err := math.Abs(r.Eval(x) - f(x))
			if err > maxErr {
				maxErr = err
				maxX = x
			}
		}

		if maxErr < bestErr {
			bestErr = maxErr
			bestR = r
		}

		// Check convergence: if the maximum error is close to the error at the
		// sample points, we are done.
		// Compute error at sample points.
		sampleErr := 0.0
		for _, x := range xs {
			err := math.Abs(r.Eval(x) - f(x))
			if err > sampleErr {
				sampleErr = err
			}
		}
		if math.Abs(maxErr-sampleErr) < 1e-12*math.Max(1, maxErr) {
			return r, maxErr, iter + 1, nil
		}

		// Exchange: replace the sample point closest to maxX with maxX.
		// Find the sample point with the smallest error.
		minErr := math.Inf(1)
		minIdx := 0
		for i, x := range xs {
			err := math.Abs(r.Eval(x) - f(x))
			if err < minErr {
				minErr = err
				minIdx = i
			}
		}
		xs[minIdx] = maxX

		// Sort xs to maintain order (important for the sign pattern).
		// We use a simple insertion sort since numPoints is small.
		for i := 1; i < len(xs); i++ {
			key := xs[i]
			j := i - 1
			for j >= 0 && xs[j] > key {
				xs[j+1] = xs[j]
				j--
			}
			xs[j+1] = key
		}
	}

	return bestR, bestErr, maxIter, errors.New("rational: minimax did not converge within maxIter")
}

// solveRemezSystem solves the linear system for the Remez algorithm.
// It returns a Rational function that interpolates the function at the
// sample points with alternating errors.
func solveRemezSystem(f func(float64) float64, xs []float64, m, n int) (*Rational, error) {
	numPoints := len(xs)
	if numPoints != m+n+2 {
		return nil, errors.New("rational: internal error: wrong number of sample points")
	}

	// Unknowns: p[0..m], q[1..n], E.
	// Total: (m+1) + n + 1 = m+n+2.
	// Equations: for each i, P(x_i) - f(x_i)*Q(x_i) = (-1)^i * E,
	// with Q(x) = 1 + sum_{j=1..n} q[j] x^j.
	// Rearranged: sum_{k=0..m} p[k] x_i^k - f_i * (1 + sum_{j=1..n} q[j] x_i^j) = (-1)^i * E.
	// => sum p[k] x_i^k - f_i * sum q[j] x_i^j - (-1)^i E = f_i.

	size := m + n + 2
	A := make([][]float64, size)
	b := make([]float64, size)
	for i := 0; i < size; i++ {
		A[i] = make([]float64, size)
		x := xs[i]
		fi := f(x)
		// Coefficients for p[0..m]
		for k := 0; k <= m; k++ {
			A[i][k] = math.Pow(x, float64(k))
		}
		// Coefficients for q[1..n]
		for j := 1; j <= n; j++ {
			A[i][m+j] = -fi * math.Pow(x, float64(j))
		}
		// Coefficient for E
		if i%2 == 0 {
			A[i][size-1] = -1
		} else {
			A[i][size-1] = 1
		}
		b[i] = fi
	}

	sol, err := solveLinear(A, b)
	if err != nil {
		return nil, err
	}

	p := sol[:m+1]
	q := make([]float64, n+1)
	q[0] = 1
	copy(q[1:], sol[m+1:m+1+n])

	return NewRational(p, q)
}

// solveLinear solves the linear system A x = b using Gaussian elimination
// with partial pivoting. A is modified in place. It returns an error if the
// matrix is singular.
func solveLinear(A [][]float64, b []float64) ([]float64, error) {
	n := len(A)
	if n == 0 {
		return nil, errors.New("rational: empty system")
	}
	if len(A[0]) != n || len(b) != n {
		return nil, errors.New("rational: dimension mismatch")
	}

	// Augment matrix with b.
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, n+1)
		copy(aug[i], A[i])
		aug[i][n] = b[i]
	}

	for col := 0; col < n; col++ {
		// Find pivot.
		pivot := col
		maxVal := math.Abs(aug[col][col])
		for r := col + 1; r < n; r++ {
			if math.Abs(aug[r][col]) > maxVal {
				maxVal = math.Abs(aug[r][col])
				pivot = r
			}
		}
		if maxVal < 1e-15 {
			return nil, errors.New("rational: singular matrix")
		}
		if pivot != col {
			aug[col], aug[pivot] = aug[pivot], aug[col]
		}

		// Eliminate.
		for r := col + 1; r < n; r++ {
			factor := aug[r][col] / aug[col][col]
			for c := col; c <= n; c++ {
				aug[r][c] -= factor * aug[col][c]
			}
		}
	}

	// Back substitution.
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := aug[i][n]
		for j := i + 1; j < n; j++ {
			sum -= aug[i][j] * x[j]
		}
		x[i] = sum / aug[i][i]
	}
	return x, nil
}
