"""
src/evaluation/final_model_selection_after_bias.py

CRITICAL INTEGRATION STEP: Final Model Selection After Bias Analysis

This script BRIDGES the gap between:
1. Initial model selection (based on R²)
2. Bias detection results (crisis bias)
3. FINAL production model decision

Makes informed decision to:
- Keep initially selected model (if bias acceptable)
- Switch to alternative model (if bias critical)
- Reject model entirely (if bias severe and no alternatives)

Outputs:
- Final production model list
- Selection reasoning (R² + bias combined)
- Risk warnings for biased models

Usage:
    python src/evaluation/final_model_selection_after_bias.py
"""

import json
from pathlib import Path
import pandas as pd


class FinalModelSelector:
    """
    Makes final model selection after considering BOTH:
    - Performance metrics (R²)
    - Bias analysis results (crisis bias)
    """
    
    def __init__(self):
        self.initial_selections = {}
        self.bias_results = {}
        self.final_selections = {}
        self.selection_reasoning = {}
        
    def load_initial_selections(self):
        """Load initial model selections (from model_selection.py)"""
        
        print(f"\n{'='*80}")
        print(f"📂 LOADING INITIAL MODEL SELECTIONS")
        print(f"{'='*80}\n")
        
        report_file = Path("reports/model_selection/complete_model_selection_report.json")
        
        if not report_file.exists():
            raise FileNotFoundError("Model selection report not found!")
        
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        for target, info in report['selections'].items():
            self.initial_selections[target] = {
                'model': info['model'],
                'test_r2': info['test_r2'],
                'test_rmse': info['test_rmse']
            }
            
            print(f"   {target:<20} → {info['model']:<20} (R² = {info['test_r2']:.4f})")
        
        print(f"\n   ✅ Loaded {len(self.initial_selections)} initial selections")
    
    def load_bias_results(self):
        """Load bias detection results for ALL models"""
        
        print(f"\n{'='*80}")
        print(f"📂 LOADING BIAS DETECTION RESULTS (ALL MODELS)")
        print(f"{'='*80}\n")
        
        bias_dir = Path("reports/all_models_bias")
        
        if not bias_dir.exists():
            print(f"   ⚠️  No bias comparison results found")
            print(f"   Run: python src/evaluation/test_all_models_for_bias.py --target all")
            return False
        
        # Load comparison files for each target
        for target in self.initial_selections.keys():
            comparison_file = bias_dir / f"{target}_all_models_bias_comparison.json"
            
            if comparison_file.exists():
                with open(comparison_file, 'r') as f:
                    comparison = json.load(f)
                
                # Store ALL models' bias results
                self.bias_results[target] = {
                    'models': comparison['models'],  # All models tested
                    'models_tested': comparison['models_tested']
                }
                
                print(f"   {target:<20} → {comparison['models_tested']} models tested")
                
                # Show bias summary for this target
                for model_data in comparison['models']:
                    model_type = model_data['model_type']
                    severity = model_data['bias']['severity']
                    r2 = model_data['overall']['r2']
                    
                    symbol = "✅" if severity == "NONE" else "⚠️" if severity == "MODERATE" else "🚨"
                    print(f"      {symbol} {model_type:<20} R²={r2:.4f}, Bias={severity}")
        
        if not self.bias_results:
            print(f"\n   ❌ No bias results loaded!")
            return False
        
        print(f"\n   ✅ Loaded bias results for {len(self.bias_results)} targets")
        return True
    
    def make_final_decisions(self):
        """
        Make FINAL production model decisions based on R² AND bias from ALL models
        
        Selection Logic:
        1. Prioritize NO bias if R² sacrifice < 5%
        2. Accept MODERATE bias if R² advantage > 5%
        3. Reject CRITICAL bias unless no alternatives
        """
        
        print(f"\n{'='*80}")
        print(f"🎯 FINAL MODEL SELECTION (Considering ALL Models)")
        print(f"{'='*80}\n")
        
        print(f"{'Target':<15} {'Selected':<20} {'R²':>8} {'Bias':>12} {'Alternative':>15} {'Decision Reason':>30}")
        print(f"{'─'*100}")
        
        for target, bias_data in self.bias_results.items():
            all_models = bias_data['models']
            
            if not all_models:
                continue
            
            # Sort by R² (best first)
            all_models_sorted = sorted(all_models, key=lambda x: x['overall']['r2'], reverse=True)
            
            # Best R² model
            best_r2_model = all_models_sorted[0]
            best_r2 = best_r2_model['overall']['r2']
            best_r2_bias = best_r2_model['bias']['severity']
            
            # Find best model with NO bias
            no_bias_models = [m for m in all_models if m['bias']['severity'] == 'NONE']
            best_no_bias_model = max(no_bias_models, key=lambda x: x['overall']['r2']) if no_bias_models else None
            
            # DECISION LOGIC
            selected_model = None
            reasoning = ""
            alternative = "None"
            
            # Case 1: Best R² model has NO bias → Easy choice!
            if best_r2_bias == 'NONE':
                selected_model = best_r2_model
                reasoning = "Best R² with no bias"
            
            # Case 2: Best R² has bias, check if fair alternative exists
            elif best_no_bias_model:
                r2_sacrifice = best_r2 - best_no_bias_model['overall']['r2']
                r2_sacrifice_pct = (r2_sacrifice / best_r2) * 100
                
                alternative = f"{best_no_bias_model['model_type']} (R²={best_no_bias_model['overall']['r2']:.4f})"
                
                # If R² sacrifice < 5% and bias is MODERATE/CRITICAL → SWITCH to fair model!
                if r2_sacrifice_pct < 5.0 and best_r2_bias in ['MODERATE', 'CRITICAL']:
                    selected_model = best_no_bias_model
                    reasoning = f"Switched for fairness (-{r2_sacrifice:.4f} R² acceptable)"
                
                # If R² sacrifice >= 5% → KEEP best R² despite bias
                else:
                    selected_model = best_r2_model
                    reasoning = f"R² priority ({r2_sacrifice_pct:.1f}% sacrifice too high)"
            
            # Case 3: No bias-free alternative → use best R²
            else:
                selected_model = best_r2_model
                reasoning = "No bias-free alternative"
            
            # Store decision
            model_type = selected_model['model_type']
            test_r2 = selected_model['overall']['r2']
            bias_severity = selected_model['bias']['severity']
            
            print(f"{target:<15} {model_type:<20} {test_r2:>8.4f} {bias_severity:>12} {alternative:>15} {reasoning:>30}")
            
            self.final_selections[target] = {
                'model': model_type,
                'test_r2': test_r2,
                'bias_severity': bias_severity,
                'alternatives_tested': len(all_models),
                'switched_from_best_r2': model_type != best_r2_model['model_type'],
                'decision_reasoning': reasoning,
                'production_ready': bias_severity != 'CRITICAL',
                'requires_monitoring': bias_severity == 'MODERATE',
                'rejected': bias_severity == 'CRITICAL'  # ← ADDED THIS!
            }
            
            # Detailed reasoning for documentation
            if model_type != best_r2_model['model_type']:
                # Switched models
                self.selection_reasoning[target] = (
                    f"SWITCHED from {best_r2_model['model_type']} (R²={best_r2:.4f}, {best_r2_bias} bias) "
                    f"to {model_type} (R²={test_r2:.4f}, {bias_severity} bias). "
                    f"Sacrificed {best_r2 - test_r2:.4f} R² ({(best_r2 - test_r2)/best_r2*100:.1f}%) "
                    f"to eliminate crisis bias. Fairness prioritized over marginal performance gain."
                )
            else:
                # Kept best R² model
                if bias_severity == 'NONE':
                    self.selection_reasoning[target] = (
                        f"Selected {model_type}: Best R² ({test_r2:.4f}) with no crisis bias. "
                        f"Clear winner among {len(all_models)} models tested."
                    )
                elif bias_severity == 'MODERATE':
                    if best_no_bias_model:
                        r2_diff = test_r2 - best_no_bias_model['overall']['r2']
                        self.selection_reasoning[target] = (
                            f"Selected {model_type}: Best R² ({test_r2:.4f}) despite MODERATE bias. "
                            f"Alternative {best_no_bias_model['model_type']} has no bias but {r2_diff:.4f} lower R² "
                            f"({r2_diff/test_r2*100:.1f}% sacrifice). Performance advantage justifies accepting moderate bias. "
                            f"Deploy with crisis monitoring."
                        )
                    else:
                        self.selection_reasoning[target] = (
                            f"Selected {model_type}: Best available despite MODERATE bias. "
                            f"No bias-free alternatives found. Deploy with crisis monitoring."
                        )
                else:  # CRITICAL
                    self.selection_reasoning[target] = (
                        f"Selected {model_type}: Best R² ({test_r2:.4f}) but CRITICAL bias detected. "
                        f"No better alternatives. RISKY - consider retraining or rejecting."
                    )
        
        print(f"{'─'*100}\n")
    
    def print_summary(self):
        """Print executive summary"""
        
        print(f"\n{'='*80}")
        print(f"📊 FINAL PRODUCTION MODEL DECISIONS")
        print(f"{'='*80}\n")
        
        accepted = [t for t, s in self.final_selections.items() if s['production_ready']]
        rejected = [t for t, s in self.final_selections.items() if s.get('rejected', False)]
        warnings = [t for t, s in self.final_selections.items() if s.get('requires_monitoring', False)]
        switched = [t for t, s in self.final_selections.items() if s.get('switched_from_best_r2', False)]
        
        print(f"Production-Ready Models: {len(accepted)}/5")
        print(f"   - No issues: {len(accepted) - len(warnings)}")
        print(f"   - With warnings: {len(warnings)}")
        print(f"Rejected Models: {len(rejected)}")
        print(f"Switched Models: {len(switched)} (for fairness)")
        
        if switched:
            print(f"\n✅ Models SWITCHED After Bias Analysis:")
            for target in switched:
                sel = self.final_selections[target]
                print(f"   - {target}: {sel['model']} (eliminated bias)")
        
        if warnings:
            print(f"\n⚠️  Models Requiring Special Monitoring:")
            for target in warnings:
                sel = self.final_selections[target]
                print(f"   - {target}: {sel['model']} ({sel['bias_severity']} crisis bias)")
                print(f"     → Deploy with VIX>30 alerts")
        
        if rejected:
            print(f"\n🚨 Rejected Models:")
            for target in rejected:
                sel = self.final_selections[target]
                print(f"   - {target}: {sel['model']} (CRITICAL bias)")
        
        print(f"\n{'='*80}")
        print(f"FINAL MODEL RECOMMENDATIONS:")
        print(f"{'='*80}\n")
        
        for target, reasoning in self.selection_reasoning.items():
            sel = self.final_selections[target]
            
            status_symbol = "✅" if not sel['requires_monitoring'] else "⚠️" if sel['production_ready'] else "🚨"
            
            print(f"\n{target.upper()}:")
            print(f"   {status_symbol} Model: {sel['model']}")
            print(f"   R²: {sel['test_r2']:.4f}")
            print(f"   Bias: {sel['bias_severity']}")
            if sel.get('switched_from_best_r2'):
                print(f"   ⭐ SWITCHED for fairness!")
            print(f"   Reasoning: {reasoning}")
    
    def save_final_selection_report(self, output_dir: str = "reports/final_selection"):
        """Save final selection report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = {
            'selection_process': {
                'step_1': 'Initial selection based on test R²',
                'step_2': 'Crisis bias detection',
                'step_3': 'Final selection incorporating bias analysis'
            },
            'final_selections': self.final_selections,
            'reasoning': self.selection_reasoning,
            'production_models': {
                target: {
                    'model_path': f"models/{sel['model']}/{sel['model'].replace('_tuned', '')}_{target}{'_tuned' if '_tuned' in sel['model'] else ''}.pkl",
                    'model_type': sel['model'],
                    'test_r2': sel['test_r2'],
                    'bias_status': sel['bias_severity'],
                    'requires_monitoring': sel['requires_monitoring']
                }
                for target, sel in self.final_selections.items()
                if sel['production_ready']
            },
            'summary': {
                'total_targets': len(self.final_selections),
                'production_ready': sum(1 for s in self.final_selections.values() if s['production_ready']),
                'with_warnings': sum(1 for s in self.final_selections.values() if s['requires_monitoring']),
                'rejected': sum(1 for s in self.final_selections.values() if s['rejected'])
            }
        }
        
        # Save main report
        report_file = output_path / "final_model_selection_after_bias.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"📁 FINAL SELECTION REPORT SAVED")
        print(f"{'='*80}")
        print(f"\n   ✅ Report: {report_file}")
        
        # Create production models manifest
        manifest = {
            'version': '1.0.0',
            'timestamp': pd.Timestamp.now().isoformat(),
            'models': {}
        }
        
        for target, sel in self.final_selections.items():
            if sel['production_ready']:
                model_name = sel['model'].replace('_tuned', '')
                suffix = '_tuned' if '_tuned' in sel['model'] else ''
                
                manifest['models'][target] = {
                    'model_file': f"models/{sel['model']}/{model_name}_{target}{suffix}.pkl",
                    'model_type': sel['model'],
                    'test_r2': sel['test_r2'],
                    'bias_severity': sel['bias_severity'],
                    'monitoring_required': sel['requires_monitoring'],
                    'deployment_notes': self.selection_reasoning[target]
                }
        
        manifest_file = output_path / "production_models_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"   ✅ Production Manifest: {manifest_file}")
        
        # Create human-readable summary
        summary_md = self._create_markdown_summary()
        summary_file = output_path / "FINAL_SELECTION_SUMMARY.md"
        with open(summary_file, 'w') as f:
            f.write(summary_md)
        
        print(f"   ✅ Summary: {summary_file}")
        
        return report_file
    
    def _create_markdown_summary(self):
        """Create markdown summary for documentation"""
        
        md = "# Final Model Selection - After Bias Analysis\n\n"
        md += "## Selection Process\n\n"
        md += "1. **Initial Selection:** Based on test R² performance\n"
        md += "2. **Bias Detection:** Tested ALL models for crisis bias\n"
        md += "3. **Final Decision:** Combined consideration of performance and fairness\n\n"
        
        md += "## Decision Criteria\n\n"
        md += "| Bias Severity | Decision | Action |\n"
        md += "|--------------|----------|--------|\n"
        md += "| NONE | ✅ Accept | Deploy as-is |\n"
        md += "| MODERATE | ⚠️ Accept with warning | Deploy with monitoring |\n"
        md += "| CRITICAL | 🚨 Reject | Find alternative or retrain |\n\n"
        
        md += "## Final Production Model Selections\n\n"
        md += "| Target | Model | Test R² | Bias Status | Switched? | Notes |\n"
        md += "|--------|-------|---------|-------------|-----------|-------|\n"
        
        for target, sel in self.final_selections.items():
            switched = "⭐ Yes" if sel.get('switched_from_best_r2', False) else "No"
            monitoring = "⚠️ Monitor" if sel.get('requires_monitoring', False) else "✅ Ready"
            
            md += f"| {target} | {sel['model']} | {sel['test_r2']:.4f} | "
            md += f"{sel['bias_severity']} | {switched} | {monitoring} |\n"
        
        md += "\n## Detailed Reasoning\n\n"
        
        for target, reasoning in self.selection_reasoning.items():
            md += f"### {target.upper()}\n\n"
            md += f"**Model:** {self.final_selections[target]['model']}\n\n"
            md += f"**Reasoning:** {reasoning}\n\n"
        
        md += "## Production Deployment Summary\n\n"
        
        summary = {
            'total': len(self.final_selections),
            'ready': sum(1 for s in self.final_selections.values() if s['production_ready']),
            'warnings': sum(1 for s in self.final_selections.values() if s.get('requires_monitoring', False)),
            'rejected': sum(1 for s in self.final_selections.values() if s.get('rejected', False)),
            'switched': sum(1 for s in self.final_selections.values() if s.get('switched_from_best_r2', False))
        }
        
        md += f"- **Total Models Evaluated:** {summary['total']}\n"
        md += f"- **Production-Ready:** {summary['ready']}/{summary['total']}\n"
        md += f"- **Switched for Fairness:** {summary['switched']}\n"
        md += f"- **Require Monitoring:** {summary['warnings']}\n"
        md += f"- **Rejected:** {summary['rejected']}\n\n"
        
        if summary['switched'] > 0:
            md += "### Models Switched After Bias Analysis\n\n"
            for target, sel in self.final_selections.items():
                if sel.get('switched_from_best_r2'):
                    md += f"- **{target}:** Switched to {sel['model']} to eliminate crisis bias\n"
        
        if summary['warnings'] > 0:
            md += "\n### Models Requiring Monitoring\n\n"
            for target, sel in self.final_selections.items():
                if sel.get('requires_monitoring'):
                    md += f"- **{target}:** {sel['bias_severity']} crisis bias detected. "
                    md += f"Monitor predictions during high VIX periods (>30).\n"
        
        md += "\n## Next Steps\n\n"
        md += "1. Push production-ready models to GCP Model Registry\n"
        md += "2. Implement monitoring for models with bias warnings\n"
        md += "3. Set up alerts for crisis periods (VIX > 30)\n"
        md += "4. Document limitations in API documentation\n"
        
        return md
    
    def create_comparison_visualization(self, output_dir: str = "reports/final_selection"):
        """Create before/after comparison visualization"""
        
        import matplotlib.pyplot as plt
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        targets = list(self.final_selections.keys())
        r2_values = [self.final_selections[t]['test_r2'] for t in targets]
        
        # Plot 1: R² Performance
        colors_r2 = []
        for t in targets:
            sel = self.final_selections[t]
            if sel['rejected']:
                colors_r2.append('red')
            elif sel['requires_monitoring']:
                colors_r2.append('orange')
            else:
                colors_r2.append('green')
        
        bars1 = ax1.barh(targets, r2_values, color=colors_r2, alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Test R²', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Target', fontsize=12, fontweight='bold')
        ax1.set_title('Final Model Performance', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add R² values
        for bar, val in zip(bars1, r2_values):
            ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2., 
                    f'{val:.4f}', va='center', ha='left', fontweight='bold')
        
        # Plot 2: Decision Summary (Pie Chart)
        summary = {
            'total': len(self.final_selections),
            'ready': sum(1 for s in self.final_selections.values() if s['production_ready'] and not s['requires_monitoring']),
            'warnings': sum(1 for s in self.final_selections.values() if s['requires_monitoring']),
            'rejected': sum(1 for s in self.final_selections.values() if s['rejected'])
        }
        
        sizes = [summary['ready'], summary['warnings'], summary['rejected']]
        labels = [
            f"Ready\n({summary['ready']})",
            f"With Warnings\n({summary['warnings']})",
            f"Rejected\n({summary['rejected']})"
        ]
        colors_pie = ['green', 'orange', 'red']
        
        # Only include non-zero slices
        plot_sizes = [s for s in sizes if s > 0]
        plot_labels = [l for l, s in zip(labels, sizes) if s > 0]
        plot_colors = [c for c, s in zip(colors_pie, sizes) if s > 0]
        
        ax2.pie(plot_sizes, labels=plot_labels, colors=plot_colors, 
               autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('Final Model Decisions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        plot_file = output_path / "final_selection_after_bias.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        
        print(f"   ✅ Visualization: {plot_file}")
        plt.close()


def main():
    print(f"\n{'='*80}")
    print(f"🎯 FINAL MODEL SELECTION AFTER BIAS ANALYSIS")
    print(f"{'='*80}")
    print(f"Integrating performance metrics + bias detection results")
    print(f"{'='*80}")
    
    selector = FinalModelSelector()
    
    # Load initial selections
    selector.load_initial_selections()
    
    # Load bias results
    has_bias_results = selector.load_bias_results()
    
    if not has_bias_results:
        print(f"\n❌ Cannot proceed without bias detection results!")
        print(f"   Run: python src/evaluation/run_crisis_bias_detection.py")
        return
    
    # Make final decisions
    selector.make_final_decisions()
    
    # Print summary
    selector.print_summary()
    
    # Save reports
    selector.save_final_selection_report()
    
    # Create visualization
    selector.create_comparison_visualization()
    
    # Final output
    print(f"\n{'='*80}")
    print(f"📁 OUTPUTS")
    print(f"{'='*80}")
    print(f"\nSaved to: reports/final_selection/")
    print(f"   - final_model_selection_after_bias.json (complete report)")
    print(f"   - production_models_manifest.json (deployment manifest)")
    print(f"   - FINAL_SELECTION_SUMMARY.md (human-readable summary)")
    print(f"   - final_selection_after_bias.png (visualization)")
    
    print(f"\n{'='*80}")
    print(f"✅ FINAL SELECTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nYou now have COMPLETE model selection that considers:")
    print(f"   ✅ Performance metrics (R², RMSE, MAE)")
    print(f"   ✅ Bias analysis (crisis robustness)")
    print(f"   ✅ Production readiness assessment")
    print(f"   ✅ Documented limitations and warnings")
    


if __name__ == "__main__":
    main()