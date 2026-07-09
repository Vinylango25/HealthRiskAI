import { Injectable, signal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class ThemeService {
  isDark = signal(true);

  toggle(): void {
    this.isDark.update(v => !v);
    document.body.classList.toggle('light', !this.isDark());
  }

  init(): void {
    const saved = localStorage.getItem('theme');
    if (saved === 'light') {
      this.isDark.set(false);
      document.body.classList.add('light');
    }
  }
}
