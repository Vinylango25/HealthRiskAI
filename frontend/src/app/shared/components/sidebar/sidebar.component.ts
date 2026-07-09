import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { NgFor, NgIf } from '@angular/common';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, NgFor, NgIf],
  templateUrl: './sidebar.component.html',
  styleUrl: './sidebar.component.scss',
})
export class SidebarComponent {
  collapsed = false;

  navItems = [
    { path: '/dashboard',      label: 'Dashboard',      icon: '📊', badge: '' },
    { path: '/insurance',      label: 'Insurance',      icon: '🏥', badge: '' },
    { path: '/credit-risk',    label: 'Credit Risk',    icon: '🏦', badge: '2' },
    { path: '/pharma',         label: 'Pharma',         icon: '💊', badge: '' },
    { path: '/simulation',     label: 'Simulation',     icon: '🎮', badge: '' },
    { path: '/explainability', label: 'Explainability', icon: '🔍', badge: '' },
    { path: '/pipeline',       label: 'Pipeline',       icon: '⚙️',  badge: '1' },
    { path: '/insights',       label: 'AI Insights',    icon: '🤖', badge: '3' },
  ];
}
