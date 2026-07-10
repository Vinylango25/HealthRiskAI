import { Component, inject, signal, HostListener } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { NgFor, NgIf } from '@angular/common';
import { ThemeService } from '../../../services/theme.service';

interface NavTab {
  label: string;
  icon: string;
  route: string;
  badge?: string;
}

@Component({
  selector: 'app-nav-tabs',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, NgFor, NgIf],
  template: `
    <header class="topnav" role="banner">

      <!-- ── Brand row ──────────────────────────────────────────────── -->
      <div class="topnav__brand">
        <span class="topnav__logo" aria-hidden="true">⚕️</span>
        <span class="topnav__title">HealthRisk AI</span>
      </div>

      <!-- ── Horizontal tabs (desktop & tablet) ────────────────────── -->
      <nav class="topnav__tabs" role="tablist" aria-label="Main navigation">
        <a
          *ngFor="let tab of tabs"
          class="topnav__tab"
          [routerLink]="tab.route"
          routerLinkActive="topnav__tab--active"
          [routerLinkActiveOptions]="{ exact: false }"
          role="tab"
          [attr.aria-label]="tab.label">
          <span class="topnav__tab-icon" aria-hidden="true">{{ tab.icon }}</span>
          <span class="topnav__tab-label">{{ tab.label }}</span>
          <span class="topnav__badge" *ngIf="tab.badge" [attr.aria-label]="tab.badge + ' notifications'">
            {{ tab.badge }}
          </span>
        </a>
      </nav>

      <!-- ── Right actions ─────────────────────────────────────────── -->
      <div class="topnav__actions">
        <span class="topnav__status" aria-label="API live">
          <span class="status-dot" aria-hidden="true"></span>
          <span class="topnav__status-text">Live</span>
        </span>
        <button
          class="topnav__btn"
          (click)="theme.toggle()"
          [attr.aria-label]="theme.isDark() ? 'Switch to light mode' : 'Switch to dark mode'">
          {{ theme.isDark() ? '☀️' : '🌙' }}
        </button>
        <div class="topnav__avatar" aria-label="User profile">VK</div>

        <!-- Hamburger: mobile only -->
        <button
          class="topnav__hamburger"
          (click)="mobileOpen.set(!mobileOpen())"
          [attr.aria-expanded]="mobileOpen()"
          aria-label="Toggle navigation menu">
          <span></span><span></span><span></span>
        </button>
      </div>
    </header>

    <!-- ── Mobile dropdown menu ───────────────────────────────────── -->
    <nav
      class="mobilenav"
      [class.mobilenav--open]="mobileOpen()"
      role="navigation"
      aria-label="Mobile navigation">
      <a
        *ngFor="let tab of tabs"
        class="mobilenav__item"
        [routerLink]="tab.route"
        routerLinkActive="mobilenav__item--active"
        [routerLinkActiveOptions]="{ exact: false }"
        (click)="mobileOpen.set(false)">
        <span class="mobilenav__icon" aria-hidden="true">{{ tab.icon }}</span>
        <span class="mobilenav__label">{{ tab.label }}</span>
        <span class="mobilenav__badge" *ngIf="tab.badge">{{ tab.badge }}</span>
      </a>
    </nav>

    <!-- Overlay to close menu when tapping outside -->
    <div
      *ngIf="mobileOpen()"
      class="mobilenav__overlay"
      (click)="mobileOpen.set(false)"
      aria-hidden="true">
    </div>
  `,
  styles: [`
    /* ── Top nav bar ───────────────────────────────────────────────────── */
    .topnav {
      position: sticky;
      top: 0;
      z-index: 200;
      height: var(--topnav-height);
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 0;
      padding: 0 20px;
      flex-shrink: 0;
    }

    /* Brand */
    .topnav__brand {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
      margin-right: 24px;
      text-decoration: none;
    }
    .topnav__logo { font-size: 22px; }
    .topnav__title {
      font-size: 15px;
      font-weight: 700;
      color: var(--text-primary);
      white-space: nowrap;
    }

    /* Tabs */
    .topnav__tabs {
      display: flex;
      align-items: stretch;
      flex: 1;
      height: 100%;
      gap: 2px;
      overflow-x: auto;
      scrollbar-width: none;
    }
    .topnav__tabs::-webkit-scrollbar { display: none; }

    .topnav__tab {
      position: relative;
      display: flex;
      align-items: center;
      gap: 7px;
      padding: 0 16px;
      height: 100%;
      font-size: 13px;
      font-weight: 500;
      color: var(--text-secondary);
      text-decoration: none;
      white-space: nowrap;
      border-bottom: 3px solid transparent;
      transition: color 0.15s, border-color 0.15s, background 0.15s;
      flex-shrink: 0;
    }
    .topnav__tab:hover {
      color: var(--text-primary);
      background: var(--bg-elevated);
    }
    .topnav__tab--active {
      color: var(--blue);
      border-bottom-color: var(--blue);
      background: rgba(26, 115, 232, 0.06);
    }
    .topnav__tab-icon { font-size: 16px; }
    .topnav__badge {
      background: var(--red);
      color: #fff;
      font-size: 10px;
      font-weight: 700;
      padding: 1px 5px;
      border-radius: 10px;
      min-width: 17px;
      text-align: center;
      line-height: 15px;
    }

    /* Right actions */
    .topnav__actions {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-left: 16px;
      flex-shrink: 0;
    }
    .topnav__status {
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 12px;
      color: var(--green);
    }
    .status-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--green);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }
    .topnav__btn {
      background: var(--bg-elevated);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 6px 10px;
      cursor: pointer;
      font-size: 14px;
      color: var(--text-primary);
      transition: background 0.15s;
    }
    .topnav__btn:hover { background: var(--border); }
    .topnav__avatar {
      width: 32px; height: 32px; border-radius: 50%;
      background: var(--blue); color: #fff;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 700;
      flex-shrink: 0;
    }

    /* Hamburger — hidden on desktop */
    .topnav__hamburger {
      display: none;
      flex-direction: column;
      gap: 5px;
      background: none;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 7px 9px;
      cursor: pointer;
    }
    .topnav__hamburger span {
      display: block;
      width: 18px; height: 2px;
      background: var(--text-primary);
      border-radius: 1px;
      transition: background 0.15s;
    }

    /* ── Mobile dropdown nav ──────────────────────────────────────────── */
    .mobilenav {
      position: fixed;
      top: var(--topnav-height);
      left: 0; right: 0;
      z-index: 199;
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border);
      box-shadow: 0 8px 24px rgba(0,0,0,0.35);
      max-height: 0;
      overflow: hidden;
      transition: max-height 0.25s ease;
    }
    .mobilenav--open {
      max-height: 400px;
    }
    .mobilenav__item {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 14px 20px;
      font-size: 15px;
      font-weight: 500;
      color: var(--text-secondary);
      text-decoration: none;
      border-left: 3px solid transparent;
      transition: background 0.15s, color 0.15s, border-color 0.15s;
    }
    .mobilenav__item:hover {
      background: var(--bg-elevated);
      color: var(--text-primary);
    }
    .mobilenav__item--active {
      color: var(--blue);
      border-left-color: var(--blue);
      background: rgba(26, 115, 232, 0.06);
    }
    .mobilenav__icon { font-size: 20px; width: 28px; text-align: center; }
    .mobilenav__label { flex: 1; }
    .mobilenav__badge {
      background: var(--red);
      color: #fff;
      font-size: 10px;
      font-weight: 700;
      padding: 1px 6px;
      border-radius: 10px;
    }
    .mobilenav__overlay {
      position: fixed;
      inset: 0;
      top: var(--topnav-height);
      z-index: 198;
      background: rgba(0,0,0,0.4);
    }

    /* ── Responsive breakpoints ──────────────────────────────────────── */

    /* Tablet (768–1024): hide tab labels, show icons only */
    @media (max-width: 1024px) and (min-width: 769px) {
      .topnav__tab-label { display: none; }
      .topnav__tab { padding: 0 14px; gap: 0; }
      .topnav__tab-icon { font-size: 18px; }
      .topnav__title { display: none; }
    }

    /* Mobile (≤768): hide horizontal tabs, show hamburger */
    @media (max-width: 768px) {
      .topnav { padding: 0 14px; }
      .topnav__tabs { display: none; }
      .topnav__hamburger { display: flex; }
      .topnav__status-text { display: none; }
      .topnav__title { font-size: 14px; }
      .topnav__btn { padding: 5px 8px; }
    }

    /* Small mobile (≤400px) */
    @media (max-width: 400px) {
      .topnav__status { display: none; }
      .topnav__title { display: none; }
    }
  `]
})
export class NavTabsComponent {
  theme = inject(ThemeService);
  mobileOpen = signal(false);

  tabs: NavTab[] = [
    { label: 'Dashboard',  icon: '📊', route: '/dashboard' },
    { label: 'Insurance',  icon: '🏥', route: '/insurance' },
    { label: 'Risk',       icon: '🏦', route: '/risk',      badge: '2' },
    { label: 'Simulation', icon: '🎮', route: '/simulation' },
    { label: 'Insights',   icon: '🤖', route: '/insights',  badge: '3' },
  ];

  /** Close mobile menu on Escape key */
  @HostListener('document:keydown.escape')
  onEscape(): void { this.mobileOpen.set(false); }
}
