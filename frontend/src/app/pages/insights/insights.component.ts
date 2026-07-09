import { Component, inject } from '@angular/core';
import { NgFor, NgClass, NgIf } from '@angular/common';
import { MockDataService } from '../../services/mock-data.service';

@Component({
  selector: 'app-insights',
  standalone: true,
  imports: [NgFor, NgClass, NgIf],
  templateUrl: './insights.component.html',
  styleUrl: './insights.component.scss',
})
export class InsightsComponent {
  private svc = inject(MockDataService);
  insights = this.svc.aiInsights();
  selected = this.insights[0];

  priorityColor(p: string): string {
    return p === 'critical' ? 'red' : p === 'high' ? 'orange' : p === 'medium' ? 'yellow' : 'blue';
  }

  categoryIcon(c: string): string {
    const m: Record<string,string> = {
      Epidemiology: '🦠', 'Credit Risk': '🏦', Pharma: '💊', Insurance: '🏥'
    };
    return m[c] ?? '🤖';
  }

  modelCards = [
    { name: 'Readmission Ensemble', auroc: 0.831, auprc: 0.573, brier: 0.119, data: 'MIMIC-IV', trained: '2026-07-01', version: 'v2.4.1' },
    { name: 'Cost Predictor (LightGBM)', auroc: null, r2: 0.28, mape: '52%', data: 'CMS Claims', trained: '2026-07-01', version: 'v1.8.0' },
    { name: 'Hospital PD Model', auroc: 0.851, gini: 0.702, ks: 0.421, data: 'CMS Hospital Compare', trained: '2026-07-05', version: 'v3.1.0' },
    { name: 'Survival Ensemble (DeepHit)', auroc: null, cindex: 0.762, data: 'MIMIC-IV Survival', trained: '2026-07-01', version: 'v1.2.0' },
  ];
}
