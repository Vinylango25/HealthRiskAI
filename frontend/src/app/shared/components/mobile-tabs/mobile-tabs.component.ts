import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { NgFor, NgIf } from '@angular/common';

interface MobileTab {
  label: string;
  icon: string;
  routes: string[];       // all routes this tab activates for
  primaryRoute: string;   // route to navigate to on tap
  badge?: string;
}

@Component({
  selector: 'app-mobile-tabs',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, NgFor, NgIf],
  template: `
    <nav class="mobile-tabs" role="tablist" aria-label="Main navigation">
      <a
        *ngFor="let tab of tabs"
        class="mobile-tabs__item"
        [routerLink]="tab.primaryRoute"
        routerLinkActive="active"
        [routerLinkActiveOptions]="{ exact: false }"
        [attr.aria-label]="tab.label"
        role="tab">
        <span class="mobile-tabs__icon">{{ tab.icon }}</span>
        <span>{{ tab.label }}</span>
        <span class="mobile-tabs__badge" *ngIf="tab.badge">{{ tab.badge }}</span>
      </a>
    </nav>
  `,
})
export class MobileTabsComponent {
  tabs: MobileTab[] = [
    {
      label: 'Overview',
      icon: '📊',
      routes: ['/dashboard'],
      primaryRoute: '/dashboard',
    },
    {
      label: 'Insurance',
      icon: '🏥',
      routes: ['/insurance'],
      primaryRoute: '/insurance',
    },
    {
      label: 'Risk',
      icon: '🏦',
      routes: ['/credit-risk', '/pharma'],
      primaryRoute: '/credit-risk',
      badge: '2',
    },
    {
      label: 'Simulate',
      icon: '🎮',
      routes: ['/simulation', '/explainability'],
      primaryRoute: '/simulation',
    },
    {
      label: 'Operations',
      icon: '⚙️',
      routes: ['/pipeline', '/insights'],
      primaryRoute: '/insights',
      badge: '3',
    },
  ];
}
