package rationalapproximation

import (
	"math"
	"testing"
)

func TestNewRational(t *testing.T) {
	p := []float64{1, 2}
	q := []float64{1, 1}
	r, err := NewRational(p, q)
	if err != nil {
		t.Fatalf("NewRational failed: %v", err)
	}
	if got := r.Eval(0); math.Abs(got-1) > 1e-12 {
		t.Errorf("Eval(0) = %v, want 1", got)
	}
	if got := r.Eval(1); math.Abs(got-1.5) > 1e-12 {
		t.Errorf("Eval(1) = %v, want 1.5", got)
	}
}

func TestNewRationalErrors(t *testing.T) {
	_, err := NewRational(nil, []float64{1})
	if err == nil {
		t.Error("expected error for empty numerator")
	}
	_, err = NewRational([]float64{1}, nil)
	if err == nil {
		t.Error("expected error for empty denominator")
	}
	_, err = NewRational([]float64{1}, []float64{1, 0})
	if err == nil {
		t.Error("expected error for zero leading denominator")
	}
}

func TestPadeExp(t *testing.T) {
	// Taylor coefficients of exp(x) around 0: 1/k!
	coeffs := []float64{1, 1, 1.0 / 2, 1.0 / 6, 1.0 / 24, 1.0 / 120}
	r, err := Pade(coeffs, 2, 2)
	if err != nil {
		t.Fatalf("Pade failed: %v", err)
	}
	// The [2/2] Padé approximant of exp(x) is (1 + x/2 + x^2/12) / (1 - x/2 + x^2/12).
	// Check at x=1: exp(1) ~ 2.71828, approximant ~ 2.7142857
	got := r.Eval(1)
	want := 2.7142857142857144
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("Eval(1) = %v, want %v", got, want)
	}
}

func TestPadeErrors(t *testing.T) {
	_, err := Pade([]float64{1}, 1, 1)
	if err == nil {
		t.Error("expected error for insufficient coefficients")
	}
	_, err = Pade([]float64{1, 2, 3}, -1, 1)
	if err == nil {
		t.Error("expected error for negative degree")
	}
}

func TestMinimaxConstant(t *testing.T) {
	f := func(x float64) float64 { return 3.0 }
	r, errVal, iter, err := Minimax(f, 0, 1, 0, 0, nil, 100)
	if err != nil {
		t.Fatalf("Minimax failed: %v", err)
	}
	if iter <= 0 {
		t.Errorf("iterations = %d, want > 0", iter)
	}
	if math.Abs(errVal) > 1e-12 {
		t.Errorf("max error = %v, want 0", errVal)
	}
	if got := r.Eval(0.5); math.Abs(got-3.0) > 1e-12 {
		t.Errorf("Eval(0.5) = %v, want 3", got)
	}
}

func TestMinimaxLinear(t *testing.T) {
	f := func(x float64) float64 { return 2*x + 1 }
	r, errVal, _, err := Minimax(f, 0, 1, 1, 0, nil, 100)
	if err != nil {
		t.Fatalf("Minimax failed: %v", err)
	}
	if errVal > 1e-12 {
		t.Errorf("max error = %v, want ~0", errVal)
	}
	if got := r.Eval(0.25); math.Abs(got-1.5) > 1e-12 {
		t.Errorf("Eval(0.25) = %v, want 1.5", got)
	}
}

func TestMinimaxRational(t *testing.T) {
	// Fit f(x) = 1/(1+x) on [0,1] with [1/1] rational.
	f := func(x float64) float64 { return 1.0 / (1.0 + x) }
	r, errVal, _, err := Minimax(f, 0, 1, 1, 1, nil, 200)
	if err != nil {
		t.Fatalf("Minimax failed: %v", err)
	}
	// The exact function is rational, so error should be tiny.
	if errVal > 1e-10 {
		t.Errorf("max error = %v, want < 1e-10", errVal)
	}
	if got := r.Eval(0.5); math.Abs(got-2.0/3.0) > 1e-10 {
		t.Errorf("Eval(0.5) = %v, want %v", got, 2.0/3.0)
	}
}

func TestMinimaxErrors(t *testing.T) {
	f := func(x float64) float64 { return x }
	_, _, _, err := Minimax(f, 1, 0, 1, 1, nil, 10)
	if err == nil {
		t.Error("expected error for invalid interval")
	}
	_, _, _, err = Minimax(nil, 0, 1, 1, 1, nil, 10)
	if err == nil {
		t.Error("expected error for nil function")
	}
	_, _, _, err = Minimax(f, 0, 1, -1, 1, nil, 10)
	if err == nil {
		t.Error("expected error for negative degree")
	}
}

func TestMinimaxNoConverge(t *testing.T) {
	// Use a function that is hard to approximate and very few iterations.
	f := func(x float64) float64 { return math.Sin(10*x) }
	_, _, iter, err := Minimax(f, 0, 1, 2, 2, nil, 1)
	if err == nil {
		t.Error("expected non-convergence error")
	}
	if iter != 1 {
		t.Errorf("iterations = %d, want 1", iter)
	}
}

func TestRationalNumDen(t *testing.T) {
	p := []float64{1, 2, 3}
	q := []float64{4, 5, 6}
	r, err := NewRational(p, q)
	if err != nil {
		t.Fatal(err)
	}
	gotP := r.Num()
	gotQ := r.Den()
	if len(gotP) != len(p) || len(gotQ) != len(q) {
		t.Fatal("length mismatch")
	}
	for i := range p {
		if gotP[i] != p[i] {
			t.Errorf("Num()[%d] = %v, want %v", i, gotP[i], p[i])
		}
	}
	for i := range q {
		if gotQ[i] != q[i] {
			t.Errorf("Den()[%d] = %v, want %v", i, gotQ[i], q[i])
		}
	}
	// Ensure copies are independent.
	gotP[0] = 99
	if r.Num()[0] == 99 {
		t.Error("Num() returned a slice that shares memory with the Rational")
	}
}
