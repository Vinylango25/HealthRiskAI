import { Component, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ThemeService } from '../../../services/theme.service';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [RouterLink],
  template: `
    <header class="topbar">
      <div class="topbar__left">
        <span class="topbar__breadcrumb">HealthRisk AI</span>
        <span class="topbar__sep">/</span>
        <span class="topbar__page">Analytics Dashboard</span>
      </div>
      <div class="topbar__right">
        <span class="topbar__status">
          <span class="status-dot"></span>
          API Live
        </span>
        <button class="topbar__btn" (click)="theme.toggle()" [title]="theme.isDark() ? 'Light mode' : 'Dark mode'">
          {{ theme.isDark() ? '☀️' : '🌙' }}
        </button>
        <div class="topbar__avatar">VK</div>
      </div>
    </header>
  `,
  styles: [`
    .topbar {
      height: var(--header-height);
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      flex-shrink: 0;
    }
    .topbar__left { display: flex; align-items: center; gap: 8px; }
    .topbar__breadcrumb { font-size: 13px; color: var(--text-muted); }
    .topbar__sep { color: var(--border-light); }
    .topbar__page { font-size: 13px; font-weight: 600; color: var(--text-primary); }
    .topbar__right { display: flex; align-items: center; gap: 12px; }
    .topbar__status {
      display: flex; align-items: center; gap: 6px;
      font-size: 12px; color: var(--green);
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
    .topbar__btn {
      background: var(--bg-elevated); border: 1px solid var(--border);
      border-radius: var(--radius-sm); padding: 6px 10px;
      cursor: pointer; font-size: 14px; color: var(--text-primary);
    }
    .topbar__avatar {
      width: 32px; height: 32px; border-radius: 50%;
      background: var(--blue); color: #fff;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 700;
    }
  `]
})
export class HeaderComponent {
  theme = inject(ThemeService);
}
