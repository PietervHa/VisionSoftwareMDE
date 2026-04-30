        const totalValue = document.getElementById("totalValue");
        const okValue = document.getElementById("okValue");
        const nokValue = document.getElementById("nokValue");
        const okRateValue = document.getElementById("okRateValue");
        const avgMsValue = document.getElementById("avgMsValue");
        const todayText = document.getElementById("todayText");
        const updatedText = document.getElementById("updatedText");
        const refreshError = document.getElementById("refreshError");

        let errorHideTimer = null;

        const hourLabels = Array.from({ length: 24 }, (_, i) => String(i));
        const okData = new Array(24).fill(0);
        const nokData = new Array(24).fill(0);
        const tabBarBtn = document.getElementById("tab-bar");
        const tabLineBtn = document.getElementById("tab-line");
        const tabNokOnlyBtn = document.getElementById("tab-nok-only");

        let nokChart = null; // line chart instance for NOK trend
        let nokOnlyChart = null; // line chart instance for NOK only

        const chart = new Chart(document.getElementById("hourlyChart"), {
            type: "bar",
            data: {
                labels: hourLabels,
                datasets: [
                    {
                        label: "OK",
                        data: okData,
                        backgroundColor: "#1e7f34",
                    },
                    {
                        label: "NOK",
                        data: nokData,
                        backgroundColor: "#9b1c1c",
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: "#eee",
                        },
                    },
                    title: {
                        display: true,
                        text: "Inspections per Hour",
                        color: "#eee",
                        font: {
                            size: 18,
                        },
                    },
                },
                scales: {
                    x: {
                        stacked: false,
                        ticks: {
                            color: "#aaa",
                        },
                        grid: {
                            color: "rgba(255,255,255,0.08)",
                        },
                        title: {
                            display: true,
                            text: "Hour",
                            color: "#aaa",
                        },
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0,
                            color: "#aaa",
                        },
                        grid: {
                            color: "rgba(255,255,255,0.08)",
                        },
                        title: {
                            display: true,
                            text: "Count",
                            color: "#aaa",
                        },
                    },
                },
            },
        });

        function formatTime(dateObj) {
            return dateObj.toLocaleTimeString([], { hour12: false });
        }

        function showRefreshError() {
            refreshError.style.visibility = "visible";
            if (errorHideTimer) {
                clearTimeout(errorHideTimer);
            }
            errorHideTimer = setTimeout(() => {
                refreshError.style.visibility = "hidden";
            }, 3000);
        }

        function setOkRateColor(rate) {
            okRateValue.classList.remove("value-ok", "value-warn", "value-nok");
            if (rate >= 90) {
                okRateValue.classList.add("value-ok");
                return;
            }
            if (rate >= 70) {
                okRateValue.classList.add("value-warn");
                return;
            }
            okRateValue.classList.add("value-nok");
        }

        function updateCards(data) {
            const total = Number(data.total || 0);
            const okCount = Number(data.ok_count || 0);
            const nokCount = Number(data.nok_count || 0);
            const okRate = Number(data.ok_rate || 0);
            const avgMs = Number(data.avg_processing_ms || 0);

            totalValue.textContent = String(total);
            okValue.textContent = String(okCount);
            nokValue.textContent = String(nokCount);
            okRateValue.textContent = `${okRate.toFixed(1)}%`;
            avgMsValue.textContent = avgMs.toFixed(2);
            setOkRateColor(okRate);

            todayText.textContent = `Today: ${data.date || new Date().toISOString().slice(0, 10)}`;
            updatedText.textContent = `Last updated: ${formatTime(new Date())}`;
        }

        function updateChart(timeline) {
            const nextOk = new Array(24).fill(0);
            const nextNok = new Array(24).fill(0);
            const source = timeline || {};

            for (let h = 0; h < 24; h += 1) {
                const key = String(h);
                const hourData = source[key] || {};
                nextOk[h] = Number(hourData.ok || 0);
                nextNok[h] = Number(hourData.nok || 0);
            }

            chart.data.datasets[0].data = nextOk;
            chart.data.datasets[1].data = nextNok;
            chart.update();
        }

        function updateNokChart(data) {
            const labels = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, '0'));
            const source = data && data.timeline ? data.timeline : {};
            const okPerHour = labels.map(h => Number((source[h] && source[h].ok) || 0));
            const nokPerHour = labels.map(h => Number((source[h] && source[h].nok) || 0));

            if (nokChart) {
                nokChart.data.labels = labels;
                nokChart.data.datasets[0].data = okPerHour;
                nokChart.data.datasets[1].data = nokPerHour;
                nokChart.update();
                return;
            }

            nokChart = new Chart(document.getElementById("chart-nok-trend"), {
                type: "line",
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: "OK per Hour",
                            data: okPerHour,
                            borderColor: "green",
                            backgroundColor: "rgba(0,200,0,0.1)",
                            fill: true,
                            tension: 0.3
                        },
                        {
                            label: "NOK per Hour",
                            data: nokPerHour,
                            borderColor: "red",
                            backgroundColor: "rgba(255,0,0,0.1)",
                            fill: true,
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { title: { display: true, text: "OK vs NOK Trend Today", color: "#eee" } },
                    scales: { y: { beginAtZero: true, ticks: { color: "#aaa" }, grid: { color: "rgba(255,255,255,0.08)" } }, x: { ticks: { color: "#aaa" }, grid: { color: "rgba(255,255,255,0.08)" } } }
                }
            });
        }

        function updateNokOnlyChart(data) {
            const labels = Array.from({ length: 24 }, (_, i) => String(i).padStart(2, '0'));
            const source = data && data.timeline ? data.timeline : {};
            const nokPerHour = labels.map(h => Number((source[h] && source[h].nok) || 0));

            if (nokOnlyChart) {
                nokOnlyChart.data.labels = labels;
                nokOnlyChart.data.datasets[0].data = nokPerHour;
                nokOnlyChart.update();
                return;
            }

            nokOnlyChart = new Chart(document.getElementById("chart-nok-only"), {
                type: "line",
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: "NOK per Hour",
                            data: nokPerHour,
                            borderColor: "#9b1c1c",
                            backgroundColor: "rgba(155,28,28,0.2)",
                            fill: true,
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: "#eee",
                            },
                        },
                        title: {
                            display: true,
                            text: "NOK Count per Hour",
                            color: "#eee",
                            font: {
                                size: 18,
                            },
                        },
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                color: "#aaa",
                            },
                            grid: {
                                color: "rgba(255,255,255,0.08)",
                            },
                        },
                        x: {
                            ticks: {
                                color: "#aaa",
                            },
                            grid: {
                                color: "rgba(255,255,255,0.08)",
                            },
                        },
                    },
                }
            });
        }

        // Tab handling: switch between bar and line charts
        function setActiveTab(tab) {
            const barCanvas = document.getElementById("hourlyChart");
            const lineCanvas = document.getElementById("chart-nok-trend");
            const nokOnlyCanvas = document.getElementById("chart-nok-only");
            
            // Hide all charts
            if (barCanvas) barCanvas.style.display = 'none';
            if (lineCanvas) lineCanvas.style.display = 'none';
            if (nokOnlyCanvas) nokOnlyCanvas.style.display = 'none';
            
            // Remove active class from all buttons
            tabBarBtn && tabBarBtn.classList.remove('active');
            tabLineBtn && tabLineBtn.classList.remove('active');
            tabNokOnlyBtn && tabNokOnlyBtn.classList.remove('active');
            
            // Show selected chart and activate button
            if (tab === 'bar') {
                if (barCanvas) barCanvas.style.display = '';
                tabBarBtn && tabBarBtn.classList.add('active');
            } else if (tab === 'line') {
                if (lineCanvas) lineCanvas.style.display = '';
                tabLineBtn && tabLineBtn.classList.add('active');
            } else if (tab === 'nok-only') {
                if (nokOnlyCanvas) nokOnlyCanvas.style.display = '';
                tabNokOnlyBtn && tabNokOnlyBtn.classList.add('active');
            }
        }

        if (tabBarBtn) {
            tabBarBtn.addEventListener('click', () => setActiveTab('bar'));
        }
        if (tabLineBtn) {
            tabLineBtn.addEventListener('click', () => setActiveTab('line'));
        }
        if (tabNokOnlyBtn) {
            tabNokOnlyBtn.addEventListener('click', () => setActiveTab('nok-only'));
        }

        async function fetchData() {
            try {
                const response = await fetch("/analytics/data", { method: "GET", cache: "no-store" });
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }

                updateCards(data);
                updateChart(data.timeline);
                updateNokChart(data);
                updateNokOnlyChart(data);
            } catch (err) {
                showRefreshError();
                updatedText.textContent = `Last updated: ${formatTime(new Date())}`;
            }
        }

        fetchData();
        setInterval(fetchData, 10000);
