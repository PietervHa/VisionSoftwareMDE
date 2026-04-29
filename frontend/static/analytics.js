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
            } catch (err) {
                showRefreshError();
                updatedText.textContent = `Last updated: ${formatTime(new Date())}`;
            }
        }

        fetchData();
        setInterval(fetchData, 10000);
