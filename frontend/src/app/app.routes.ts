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
    path: 'risk',
    loadComponent: () =>
      import('./pages/risk/risk.component').then(m => m.RiskComponent),
  },
  {
    path: 'simulation',
    loadComponent: () =>
      import('./pages/simulation/simulation.component').then(m => m.SimulationComponent),
  },
  {
    path: 'insights',
    loadComponent: () =>
      import('./pages/insights/insights.component').then(m => m.InsightsComponent),
  },
  { path: '**', redirectTo: 'dashboard' },
];
