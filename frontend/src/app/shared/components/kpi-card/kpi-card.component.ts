import { Component, Input } from '@angular/core';
import { NgClass } from '@angular/common';
import { KpiCard } from '../../../services/mock-data.service';

@Component({
  selector: 'app-kpi-card',
  standalone: true,
  imports: [NgClass],
  template: `
    <div class="kpi-card" [title]="kpi.description">
      <div class="kpi-card__accent" [style.background]="kpi.color"></div>
      <div class="kpi-card__icon" [style.background]="kpi.color + '22'" [style.color]="kpi.color">
        {{ kpi.icon }}
      </div>
      <div class="kpi-card__label">{{ kpi.label }}</div>
      <div class="kpi-card__value">{{ kpi.value }}</div>
      <div class="kpi-card__delta" [ngClass]="kpi.deltaDir">
        <span>{{ kpi.deltaDir === 'pos' ? '▲' : kpi.deltaDir === 'neg' ? '▼' : '●' }}</span>
        {{ kpi.delta }}
      </div>
    </div>
  `,
})
export class KpiCardComponent {
  @Input() kpi!: KpiCard;
}
