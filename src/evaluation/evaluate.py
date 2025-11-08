import pprint

def summarize_results(all_results):
    # all_results[backbone][model_name] = metrics dict
    print("\n=== Comparative report (macro metrics) ===")
    for backbone, models in all_results.items():
        print(f"\nBackbone: {backbone}")
        for mname, metrics in models.items():
            print(f"  {mname}: "
                  f"Acc={metrics['Accuracy']:.4f}, "
                  f"Prec={metrics['Precision']:.4f}, "
                  f"Rec={metrics['Recall']:.4f}, "
                  f"F1={metrics['F1']:.4f}")
    print("\nDetailed confusion matrices available in results objects.")
