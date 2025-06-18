const { DateTime } = luxon;
const canvas = document.getElementById('exampleChart');
canvas.classList.add('bg-black');
canvas.parentElement.classList.add('chart-wrapper');
const hoverInfo = document.getElementById('hover-info');
function generarSerie (periodo) {
  const etiquetas = [];
  const datos     = [];
  const hoy       = DateTime.now();
  const push = (fecha, fmt, base, sinF, noise) => {
    etiquetas.push(fecha.toFormat(fmt));
    const valor = base + Math.sin(sinF) * base * 0.12 + (Math.random() - 0.5) * noise;
    datos.push(Number(valor.toFixed(2)));
  };
  switch (periodo) {
    case 'semanal': {
      for (let i = 51; i >= 0; i--) {
        push(hoy.minus({ weeks: i }), "'W'W", 100, (51 - i) / 4, 6);
      }
      break;
    }
    case 'mensual': {
      for (let i = 23; i >= 0; i--) {
        push(hoy.minus({ months: i }), 'LLL yy', 100, (23 - i) / 3, 8);
      }
      break;
    }
    default: {
      for (let i = 9; i >= 0; i--) {
        push(hoy.minus({ years: i }), 'yyyy', 100, (9 - i) / 1.5, 12);
      }
    }
  }
  return { etiquetas, datos };
}
function extremaData (datos, labels) {
  const max = Math.max(...datos);
  const min = Math.min(...datos);
  return [
    { x: labels[datos.indexOf(max)], y: max },
    { x: labels[datos.indexOf(min)], y: min }
  ];
}
let grafico = null;
function initGrafico (periodoInicial = 'semanal') {
  const { etiquetas, datos } = generarSerie(periodoInicial);
  const datasetLinea = {
    label: 'Índice de Reservas',
    data: datos,
    tension: 0.35,
    borderWidth: 3,
    pointRadius: 0,
    segment: {
      borderColor: ctx => (ctx.p1.parsed.y >= ctx.p0.parsed.y ? '#22c55e' : '#ef4444'),
      borderDash:  ctx => (ctx.p1.parsed.y >= ctx.p0.parsed.y ? undefined      : [5, 4])
    },
    fill: {
      target: 'origin',
      above: 'rgba(34,197,94,0.15)',
      below: 'rgba(239,68,68,0.15)'
    }
  };
  const datasetPuntos = {
    type: 'scatter',
    label: 'Extremos',
    data: extremaData(datos, etiquetas),
    pointRadius: 5,
    pointStyle: ['triangle', 'rectRot'],
    backgroundColor: ['#22c55e', '#ef4444'],
    borderColor: '#000',
    borderWidth: 1
  };
  grafico = new Chart(canvas, {
    type: 'line',
    data: {
      labels: etiquetas,
      datasets: [datasetLinea, datasetPuntos]
    },
    options: opcionesChart(),
    plugins: []
  });
  actualizarDescripcion(periodoInicial, datos);
}
function opcionesChart () {
  return {
    responsive: true,
    animation: { duration: 600 },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1f2937',
        borderColor:     '#374151',
        borderWidth: 1,
        titleColor: '#f3f4f6',
        bodyColor:  '#e5e7eb',
        callbacks: {
          label: ctx => `Valor: ${ctx.parsed.y}`
        }
      }
    },
    onHover: (event, elements) => {
      if (!hoverInfo) return;
      if (elements.length && elements[0].datasetIndex === 0) { 
        const idx   = elements[0].index;
        const label = grafico.data.labels[idx];
        const val   = grafico.data.datasets[0].data[idx];

        hoverInfo.innerHTML = `<span class="font-semibold">${label}</span>: ${val}`;
      } else {
        hoverInfo.textContent = '';
      }
    },
    scales: {
      x: {
        ticks: { color: '#9ca3af', maxRotation: 60, minRotation: 60 },
        grid:  { display: false }
      },
      y: {
        ticks: { color: '#9ca3af' },
        grid:  { color: '#374151' }
      }
    }
  };
}
function actualizarDescripcion (periodo, datos) {
  const descripcion = document.getElementById('chart-description');
  const diff        = datos.at(-1) - datos[0];
  const pct         = ((diff / datos[0]) * 100).toFixed(1);
  const tendencia   = diff >= 0 ? 'subida' : 'bajada';
  const color       = diff >= 0 ? '#22c55e' : '#ef4444';
  const textoPeriodo = periodo === 'semanal' ? 'semanal' : periodo === 'mensual' ? 'mensual' : 'anual';
  descripcion.innerHTML = `El índice muestra una <span style="color:${color};font-weight:600">${tendencia} de ${Math.abs(pct)} %</span> en la tendencia ${textoPeriodo} más reciente.`;
}
const botones = document.querySelectorAll('.period-btn');
botones.forEach(btn => {
  btn.addEventListener('click', e => {
    botones.forEach(b => b.classList.remove('bg-gray-800', 'text-white'));
    e.currentTarget.classList.add('bg-gray-800', 'text-white');
    const periodo              = e.currentTarget.dataset.period;
    const { etiquetas, datos } = generarSerie(periodo);
    grafico.data.labels          = etiquetas;
    grafico.data.datasets[0].data = datos;
    grafico.data.datasets[1].data = extremaData(datos, etiquetas);
    grafico.update();
    actualizarDescripcion(periodo, datos);
  });
});
initGrafico('semanal');
