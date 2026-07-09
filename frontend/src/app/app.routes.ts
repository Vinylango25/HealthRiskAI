import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
  {
    path: 'dashboard',
    loadComponent: () =>
      import('./pages/dashboard/dashboard.component').then(m => m.DashboardComponent),
  },
  {
    path: 'insurance',
    loadComponent: () =>
      import('./pages/insurance/insurance.component').then(m => m.InsuranceComponent),
  },
  {
    path: 'credit-risk',
    loadComponent: () =>
      import('./pages/credit-risk/credit-risk.component').then(m => m.CreditRiskComponent),
  },
  {
    path: 'pharma',
    loadComponent: () =>
      import('./pages/pharma/pharma.component').then(m => m.PharmaComponent),
  },
  {
    path: 'simulation',
    loadComponent: () =>
      import('./pages/simulation/simulation.component').then(m => m.SimulationComponent),
  },
  {
    path: 'explainability',
    loadComponent: () =>
      import('./pages/explainability/explainability.component').then(m => m.ExplainabilityComponent),
  },
  {
    path: 'pipeline',
    loadComponent: () =>
      import('./pages/pipeline/pipeline.component').then(m => m.PipelineComponent),
  },
  {
    path: 'insights',
    loadComponent: () =>
      import('./pages/insights/insights.component').then(m => m.InsightsComponent),
  },
  { path: '**', redirectTo: 'dashboard' },
];
