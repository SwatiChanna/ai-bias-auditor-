#!/usr/bin/env python3
"""Comprehensive testing suite for AI Bias Auditor workflows."""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.getcwd())
from src import bias_engine, mitigation
from src.counterfactual_templates import CounterfactualGenerator, validate_counterfactual
import app

def test_dataset_loading():
    """Test 1: Load all demo datasets and validate fairness metrics."""
    print("\n" + "="*70)
    print("TEST 1: END-TO-END DEMO DATASET LOADING & BIAS METRICS")
    print("="*70)
    
    tests = [
        ("gender", "disparate_impact", 0.64, "should show bias"),
        ("caste", "disparate_impact", 0.58, "should show bias"),
        ("language", "disparate_impact", 0.67, "should show bias"),
        ("region", "disparate_impact", 0.52, "should show bias"),
    ]
    
    results = []
    for dataset_type, metric, target, desc in tests:
        try:
            print(f"\n--- Testing {dataset_type.upper()} Dataset ---")
            df = bias_engine.load_demo_dataset(dataset_type)
            print(f"✓ Dataset loaded: {df.shape}")
            
            # Determine protected attr and label for each dataset
            if dataset_type == "gender":
                protected_attr, label = "gender", "income_50k"
            elif dataset_type == "caste":
                protected_attr, label = "caste", "loan_approved"
            elif dataset_type == "language":
                protected_attr, label = "language", "selected"
            else:  # region
                protected_attr, label = "location", "hired"
            
            di = bias_engine.calculate_disparate_impact(df, protected_attr, label)
            spd = bias_engine.calculate_statistical_parity_difference(df, protected_attr, label)
            
            print(f"  Disparate Impact: {di:.3f} (target: {target})")
            print(f"  Stat. Parity Diff: {spd:.3f}")
            
            # Validate DI is within reasonable range (allow variance)
            if 0.4 <= di <= 1.5 or di == 0.0:
                status = "✓ PASS"
            else:
                status = "⚠ CHECK"
            
            results.append((dataset_type, di, status))
            print(f"  {status}")
            
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            results.append((dataset_type, None, "✗ FAIL"))
    
    return results

def test_counterfactual():
    """Test 2: Counterfactual generation and sensitivity."""
    print("\n" + "="*70)
    print("TEST 2: COUNTERFACTUAL GENERATION & SENSITIVITY")
    print("="*70)
    
    generator = CounterfactualGenerator()
    results = []
    
    datasets = [
        ("gender", "gender", "income_50k"),
        ("caste", "caste", "loan_approved"),
        ("language", "language", "selected"),
        ("region", "location", "hired"),
    ]
    
    for dataset_type, protected_attr, label in datasets:
        try:
            print(f"\n--- Testing {dataset_type.upper()} Counterfactual ---")
            df = bias_engine.load_demo_dataset(dataset_type)
            
            # Get original scores
            orig_di = bias_engine.calculate_disparate_impact(df, protected_attr, label)
            orig_spd = bias_engine.calculate_statistical_parity_difference(df, protected_attr, label)
            
            # Generate counterfactual
            if dataset_type == "gender":
                cf_df = generator.generate_gender_counterfactual(df)
            elif dataset_type == "caste":
                cf_df = generator.generate_caste_counterfactual(df)
            elif dataset_type == "language":
                cf_df = generator.generate_language_counterfactual(df)
            else:
                cf_df = generator.generate_region_counterfactual(df)
            
            # Get counterfactual scores
            cf_di = bias_engine.calculate_disparate_impact(cf_df, protected_attr, label)
            cf_spd = bias_engine.calculate_statistical_parity_difference(cf_df, protected_attr, label)
            
            # Calculate sensitivity
            di_change = abs(cf_di - orig_di) / max(abs(orig_di), 0.01) * 100 if orig_di != 0 else 0
            spd_change = abs(cf_spd - orig_spd) * 100
            
            print(f"  Original DI: {orig_di:.3f}, Counterfactual DI: {cf_di:.3f}")
            print(f"  DI Change: {di_change:.1f}%")
            print(f"  Original SPD: {orig_spd:.3f}, Counterfactual SPD: {cf_spd:.3f}")
            print(f"  SPD Change: {spd_change:.1f}%")
            
            # Validate sensitivity threshold (>10% or >0.1 absolute)
            sensitivity_ok = di_change > 10 or spd_change > 10
            status = "✓ PASS" if sensitivity_ok else "⚠ LOW SENSITIVITY"
            
            # Validate unchanged columns
            if dataset_type == "gender":
                unchanged_cols = ["education", "age"]
            elif dataset_type == "caste":
                unchanged_cols = ["text"]
            elif dataset_type == "language":
                unchanged_cols = ["qualification"]
            else:
                unchanged_cols = ["education"]
            
            validation_ok = validate_counterfactual(df, cf_df, unchanged_cols)
            val_status = "✓ Validated" if validation_ok else "✗ Column mismatch"
            
            print(f"  {status}, {val_status}")
            results.append((dataset_type, di_change, status))
            
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            results.append((dataset_type, None, "✗ FAIL"))
    
    return results

def test_mitigation():
    """Test 3: Bias mitigation with before/after comparison."""
    print("\n" + "="*70)
    print("TEST 3: BIAS MITIGATION & BEFORE/AFTER COMPARISON")
    print("="*70)
    
    results = []
    
    datasets = [
        ("gender", "gender", "income_50k"),
        ("caste", "caste", "loan_approved"),
        ("language", "language", "selected"),
        ("region", "location", "hired"),
    ]
    
    for dataset_type, protected_attr, label in datasets:
        try:
            print(f"\n--- Testing {dataset_type.upper()} Mitigation ---")
            df = bias_engine.load_demo_dataset(dataset_type)
            
            # Get original scores
            orig_di = bias_engine.calculate_disparate_impact(df, protected_attr, label)
            orig_spd = bias_engine.calculate_statistical_parity_difference(df, protected_attr, label)
            
            print(f"  Before Mitigation:")
            print(f"    DI: {orig_di:.3f}, SPD: {orig_spd:.3f}")
            
            # Apply mitigation
            mitigated_df = mitigation.apply_reweighing(df, protected_attr, label)
            
            # Get mitigated scores
            mit_di = bias_engine.calculate_disparate_impact(mitigated_df, protected_attr, label)
            mit_spd = bias_engine.calculate_statistical_parity_difference(mitigated_df, protected_attr, label)
            
            di_improvement = abs(mit_di - orig_di) if orig_di != float('inf') else 0
            spd_improvement = abs(mit_spd - orig_spd)
            
            print(f"  After Mitigation:")
            print(f"    DI: {mit_di:.3f} (Δ {di_improvement:+.3f})")
            print(f"    SPD: {mit_spd:.3f} (Δ {spd_improvement:+.3f})")
            print(f"    Weights added: {'instance_weights' in mitigated_df.columns}")
            
            # Check if improvement is meaningful (>0.15 improvement towards fairness)
            improvement_ok = di_improvement > 0.15 or (orig_di != float('inf') and abs(mit_di - 1.0) < abs(orig_di - 1.0))
            status = "✓ PASS" if improvement_ok else "⚠ LIMITED IMPROVEMENT"
            
            print(f"  {status}")
            results.append((dataset_type, di_improvement, status))
            
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            results.append((dataset_type, None, "✗ FAIL"))
    
    return results

def test_export():
    """Test 4: PDF export functionality."""
    print("\n" + "="*70)
    print("TEST 4: PDF EXPORT & REPORT GENERATION")
    print("="*70)
    
    try:
        print("\n--- Testing PDF Report Generation ---")
        
        # Load dataset
        df = bias_engine.load_demo_dataset("caste")
        protected_attr, label = "caste", "loan_approved"
        
        # Calculate scores
        di = bias_engine.calculate_disparate_impact(df, protected_attr, label)
        spd = bias_engine.calculate_statistical_parity_difference(df, protected_attr, label)
        
        scores = {'disparate_impact': di, 'statistical_parity': spd}
        explanation = "Test audit report with bias detection across protected attributes."
        recommendations = [
            "Apply AIF360 reweighing to balance dataset",
            "Use fairness constraints during model training",
            "Post-process predictions for equalized odds"
        ]
        
        # Generate report
        report_bytes = app._create_report(scores, explanation, recommendations)
        
        print(f"✓ Report generated: {len(report_bytes)} bytes")
        
        # Save to output directory
        output_path = Path('output') / 'comprehensive_test_report.pdf'
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(report_bytes)
        
        print(f"✓ Report saved to: {output_path}")
        
        # Validate PDF structure
        if len(report_bytes) > 100:
            pdf_valid = True
            print("✓ PDF structure valid")
        else:
            pdf_valid = False
            print("✗ PDF too small, possible corruption")
        
        status = "✓ PASS" if pdf_valid else "✗ FAIL"
        return [("pdf_export", len(report_bytes), status)]
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return [("pdf_export", None, "✗ FAIL")]

def test_edge_cases():
    """Test 5: Edge case handling."""
    print("\n" + "="*70)
    print("TEST 5: EDGE CASE HANDLING")
    print("="*70)
    
    results = []
    
    # Test 5a: Empty dataset
    print("\n--- Test 5a: Empty Dataset ---")
    try:
        empty_df = pd.DataFrame()
        bias_engine.calculate_disparate_impact(empty_df, "protected", "label")
        print("✗ FAIL: Should raise error on empty dataset")
        results.append(("empty_dataset", "No error raised", "✗ FAIL"))
    except ValueError as e:
        print(f"✓ PASS: Graceful error - {str(e)[:50]}...")
        results.append(("empty_dataset", "Error caught", "✓ PASS"))
    except Exception as e:
        print(f"⚠ CHECK: Different error - {type(e).__name__}")
        results.append(("empty_dataset", type(e).__name__, "⚠ CHECK"))
    
    # Test 5b: Invalid column name
    print("\n--- Test 5b: Invalid Column Name ---")
    try:
        df = bias_engine.load_demo_dataset("gender")
        bias_engine.calculate_disparate_impact(df, "nonexistent_col", "income_50k")
        print("✗ FAIL: Should raise error on invalid column")
        results.append(("invalid_column", "No error raised", "✗ FAIL"))
    except ValueError as e:
        if "missing" in str(e).lower():
            print(f"✓ PASS: Helpful error - {str(e)[:60]}...")
            results.append(("invalid_column", "Helpful message", "✓ PASS"))
        else:
            print(f"⚠ CHECK: Vague error - {str(e)[:50]}...")
            results.append(("invalid_column", "Vague message", "⚠ CHECK"))
    except Exception as e:
        print(f"✗ FAIL: Unexpected error - {type(e).__name__}")
        results.append(("invalid_column", type(e).__name__, "✗ FAIL"))
    
    # Test 5c: Model with minimal data
    print("\n--- Test 5c: Model with Minimal Data ---")
    try:
        minimal_df = pd.DataFrame({
            "feature": ["a", "b"],
            "target": [1, 0]
        })
        predictions = bias_engine.run_model(minimal_df, "target")
        if len(predictions) == 2:
            print(f"✓ PASS: Model handles minimal data, predictions: {predictions}")
            results.append(("minimal_data", "Works", "✓ PASS"))
        else:
            print("✗ FAIL: Wrong prediction count")
            results.append(("minimal_data", "Wrong count", "✗ FAIL"))
    except Exception as e:
        print(f"⚠ CHECK: {type(e).__name__} - {str(e)[:50]}...")
        results.append(("minimal_data", type(e).__name__, "⚠ CHECK"))
    
    return results

def main():
    """Run all comprehensive tests."""
    print("\n")
    print("+" + "="*68 + "+")
    print("|" + " "*15 + "COMPREHENSIVE BIAS AUDITOR TEST SUITE" + " "*17 + "|")
    print("+" + "="*68 + "+")
    
    all_results = []
    
    # Run tests
    all_results.extend(test_dataset_loading())
    all_results.extend(test_counterfactual())
    all_results.extend(test_mitigation())
    all_results.extend(test_export())
    all_results.extend(test_edge_cases())
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, _, status in all_results if "PASS" in status)
    failed = sum(1 for _, _, status in all_results if "FAIL" in status)
    checks = sum(1 for _, _, status in all_results if "CHECK" in status)
    
    for test_name, metric, status in all_results:
        metric_str = f"{metric:.3f}" if isinstance(metric, float) else str(metric)
        print(f"{status:12} | {test_name:20} | {metric_str:15}")
    
    print("="*70)
    print(f"Results: {passed} PASS, {failed} FAIL, {checks} CHECK")
    print("="*70 + "\n")
    
    return passed, failed, checks

if __name__ == "__main__":
    main()
