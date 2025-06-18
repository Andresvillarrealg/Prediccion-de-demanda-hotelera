document.addEventListener("DOMContentLoaded", function () {
  fetch('/api/forecast')
    .then(response => response.json())
    .then(datosForecast => {
      const predicciones = datosForecast.map(d => parseFloat(d.prediccion_lgb));
      const promedio = Math.round(predicciones.reduce((a, b) => a + b, 0) / predicciones.length);
      const max = Math.round(Math.max(...predicciones));
      const min = Math.round(Math.min(...predicciones));
      const maxObj = datosForecast.reduce((a, b) => parseFloat(a.prediccion_lgb) > parseFloat(b.prediccion_lgb) ? a : b);
      const pico = `${maxObj.fecha}`;
      const diasAltos = datosForecast.filter(d => parseFloat(d.prediccion_lgb) > 900).length;
      const finSemana = datosForecast.filter(d => d.es_viernes_o_sabado == 1 || d.es_viernes_o_sabado == "1");
      const promedioFinSemana = finSemana.length ? Math.round(finSemana.reduce((a, b) => a + parseFloat(b.prediccion_lgb), 0)/finSemana.length) : "N/A";
      const diasFestivos = datosForecast.filter(d => d.es_festivo == 1 || d.es_festivo == "1").length;
      const diasTemporada = datosForecast.filter(d => d.temporada_alta == 1 || d.temporada_alta == "1").length;
      const kpis = [
        {id: "promedio", value: promedio},
        {id: "max", value: max},
        {id: "min", value: min},
        {id: "pico", value: pico},
        {id: "altos", value: diasAltos},
        {id: "finsemana", value: promedioFinSemana !== null ? promedioFinSemana : "N/A"},
        {id: "festivos", value: diasFestivos > 0 ? diasFestivos : "Sin festivos"},
        {id: "temporada", value: diasTemporada > 0 ? diasTemporada : "Sin temporada alta"}
      ];
      kpis.forEach((kpi, idx) => {
        setTimeout(() => {
          document.getElementById("kpi-" + kpi.id).textContent = kpi.value;
          const card = document.getElementById("kpi-card-" + kpi.id);
          if(card){
            card.classList.remove("opacity-0", "translate-y-4");
            card.classList.add("opacity-100", "translate-y-0");
          }
        }, 80 * idx);
      });
      const ctxForecast = document.getElementById('forecastChart').getContext('2d');
      new Chart(ctxForecast, {
        type: 'line',
        data: {
          labels: datosForecast.map(d => d.fecha),
          datasets: [{
            label: 'Ocupación pronosticada',
            data: predicciones,
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239,68,68,0.10)',
            tension: 0.3,
            fill: true,
            pointRadius: 2
          }]
        },
        options: {
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: true } }
        }
      });
      const ctxDiasClave = document.getElementById('diasClaveChart').getContext('2d');
      new Chart(ctxDiasClave, {
        type: 'bar',
        data: {
          labels: ['Festivo', 'Viernes/Sábado', 'Temporada Alta', 'Alta Demanda (>900)'],
          datasets: [{
            label: 'Días',
            data: [
              diasFestivos,
              finSemana.length,
              diasTemporada,
              diasAltos
            ],
            backgroundColor: ['#f59e42', '#0ea5e9', '#10b981', '#e11d48']
          }]
        },
        options: {
          plugins: { legend: { display: false } },
          indexAxis: 'y',
          scales: { x: { beginAtZero: true, precision: 0 } }
        }
      });
      const tabla = document.getElementById("tablaPronostico").querySelector("tbody");
      tabla.innerHTML = datosForecast.slice(0, 10).map(d => `
        <tr>
          <td class="px-4 py-2">${d.fecha}</td>
          <td class="px-4 py-2">${Math.round(d.prediccion_lgb)}</td>
          <td class="px-4 py-2">${(d.es_festivo == 1 || d.es_festivo == "1") ? "Sí" : "No"}</td>
          <td class="px-4 py-2">${(d.es_viernes_o_sabado == 1 || d.es_viernes_o_sabado == "1") ? "Sí" : "No"}</td>
          <td class="px-4 py-2">${(d.temporada_alta == 1 || d.temporada_alta == "1") ? "Sí" : "No"}</td>
        </tr>
      `).join('');
    });
});
