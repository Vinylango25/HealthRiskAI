import { Component, OnInit, inject, signal } from '@angular/core';
import { NgFor, NgClass } from '@angular/common';
import { NgApexchartsModule } from 'ng-apexcharts';
import { MockDataService } from '../../services/mock-data.service';

@Component({
  selector: 'app-simulation',
  standalone: true,
  imports: [NgFor, NgClass, NgApexchartsModule],
  templateUrl: './simulation.component.html',
  styleUrl: './simulation.component.scss',
})
export class SimulationComponent implements OnInit {
  private svc = inject(MockDataService);

  state = this.svc.simulationState();
  running = signal(false);
  selectedScenario = 'pandemic';

  equityChartOptions: any = {};
  scoreChartOptions: any = {};

  scenarios = [
    { id:'pandemic',  label:'Pandemic Outbreak',      icon:'🦠', severity:'Severe',   impact:'−12% portfolio, ICU surge, claims spike' },
    { id:'drug',      label:'Drug Safety Withdrawal',  icon:'💊', severity:'Moderate', impact:'−40–80% pharma equity, FAERS trigger' },
    { id:'cms',       label:'CMS Rate Cut',            icon:'🏛️', severity:'Moderate', impact:'−3–5% Medicare revenue per hospital' },
    { id:'merger',    label:'Hospital Merger',         icon:'🏥', severity:'Low',      impact:'+8% operating efficiency, spread tightening' },
    { id:'rate',      label:'Interest Rate Shock',     icon:'📈', severity:'Moderate', impact:'Bond duration risk, +150 bps curve shift' },
    { id:'cyber',     label:'Cyber Attack',            icon:'💻', severity:'Severe',   impact:'$50M liability, 3-week downtime scenario' },
  ];

  decisions = [
    { id:'d1', label:'Increase IBNR Reserves +15%', icon:'📦', cost:'$3.6M',  effect:'Pandemic protection' },
    { id:'d2', label:'Reduce Hospital Bond Duration', icon:'⏱️', cost:'−0.4% yield', effect:'Rate shock hedge' },
    { id:'d3', label:'Increase Pharma Allocation',   icon:'💊', cost:'+$22M',  effect:'Pipeline upside' },
    { id:'d4', label:'Buy Portfolio Protection',     icon:'🛡️', cost:'$1.8M premium', effect:'Tail-risk hedge' },
    { id:'d5', label:'Activate Early Warning Alert', icon:'🚨', cost:'—',     effect:'R₀ monitoring trigger' },
    { id:'d6', label:'Rebalance to Defensives',      icon:'⚖️', cost:'$0.5M transaction', effect:'Reduce beta' },
  ];

  ngOnInit(): void {
    this.buildEquityChart();
    this.buildScoreChart();
  }

  advanceQuarter(): void {
    this.running.set(true);
    setTimeout(() => {
      this.state = { ...this.state, quarter: this.state.quarter + 1,
        portfolioValue: +(this.state.portfolioValue * 1.032).toFixed(1),
        score: { ...this.state.score, total: Math.min(1000, this.state.score.total + 18) }
      };
      this.running.set(false);
    }, 1200);
  }

  private buildEquityChart(): void {
    const curves = this.svc.portfolioEquityCurve();
    this.equityChartOptions = {
      series: [
        { name: 'AI Opponent',   data: curves[0].slice(0, this.state.quarter + 1).map(p => p.y) },
        { name: 'Your Portfolio', data: curves[1].slice(0, this.state.quarter + 1).map(p => p.y) },
        { name: 'Benchmark',     data: curves[2].slice(0, this.state.quarter + 1).map(p => p.y) },
      ],
      chart: { type: 'line', height: 260, background: 'transparent', toolbar: { show: false } },
      stroke: { curve: 'smooth', width: [3,2,1.5], dashArray: [0,0,4] },
      colors: ['#1a73e8','#34a853','#8b949e'],
      xaxis: { categories: curves[0].slice(0, this.state.quarter + 1).map(p => p.x),
               labels: { style: { colors: '#8b949e', fontSize: '10px' } } },
      yaxis: { labels: { formatter: (v: number) => `$${v.toFixed(0)}M`, style: { colors: '#8b949e' } } },
      tooltip: { theme: 'dark', y: { formatter: (v: number) => `$${v.toFixed(1)}M` } },
      legend: { labels: { colors: '#8b949e' } },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      annotations: { xaxis: [{ x: 'Q8', borderColor: '#ea4335',
        label: { text: '🦠 Pandemic', style: { color: '#ea4335', background: '#21262d', fontSize: '10px' } } }] },
      theme: { mode: 'dark' },
    };
  }

  private buildScoreChart(): void {
    const s = this.state.aiScore;
    this.scoreChartOptions = {
      series: [s.performance, s.risk, s.clinical, s.speed],
      chart: { type: 'donut', height: 200, background: 'transparent' },
      labels: ['Performance (400)', 'Risk Mgmt (300)', 'Clinical Intel (200)', 'Speed (100)'],
      colors: ['#1a73e8','#a855f7','#34a853','#f9c74f'],
      plotOptions: { pie: { donut: { size: '60%',
        labels: { show: true, total: { show: true, label: 'AI Score',
          color: '#8b949e', formatter: () => `${s.total}/1000` } } } } },
      legend: { labels: { colors: '#8b949e' }, fontSize: '10px', position: 'bottom' },
      tooltip: { theme: 'dark' },
      theme: { mode: 'dark' },
    };
  }
}
