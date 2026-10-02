import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def estimate_pi(N, rng):
    x = rng.random(N)
    y = rng.random(N)
    inside = x**2 + y**2 <= 1
    return 4 * inside.mean()

rng = np.random.default_rng(42)
for N in [100, 1000, 10000, 100000, 1000000]:
  est = estimate_pi(N, rng)
  print(f"N = {N:>9,}   π ≈ {est:.5f}   오차 = {abs(est - np.pi):.5f}")

"""MC로 pi값 구하기"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N = norm.cdf

def CallOptionBS(S, K, T, r, q, sigma):
    if T == 0:
        return np.max(S - K, 0)
    else:
        d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)

def call_option_by_MC():
    s0 = 100 # 기초자산의 현재가
    vol = 0.3 # 기초자산의 변동성
    r = 0.02 # 무위험이자율
    d = 0.01 # 기초자산의 연속배당률
    maturity = 1 # 콜옵션의 만기
    strike = 100 # 콜옵션의 행사가

    nSimulation = 1000000 # 시뮬레이션 몇번 돌릴건지
    drift = (r - d - 0.5 * vol ** 2) * maturity
    volsqrtmat = vol * np.sqrt(maturity)

    rn = np.random.normal(size=nSimulation)
    s_maturity = s0 * np.exp(drift + volsqrtmat * rn)
    # 표준정규분포 난수를 발생시켜서 만기 종가를 생성하기
    payoff = np.array([np.max([0, s_mat - strike]) for s_mat in s_maturity])
    # 종가 각각에 대해 콜옵선 payoff를 게산해서 배열에 집어넣기

    s_cum_mean = np.cumsum(payoff) / np.arange(1, nSimulation + 1) * np.exp(-r * maturity)
    # cumsum : k항까지의 평균 -> n을 증가할수록 어떻게 수렴하는지 보기
    simulation_result = np.mean(payoff) * np.exp(-r * maturity)
    exact_solution = CallOptionBS(s0, strike, maturity, r, d, vol)

    print('result of simulation :{:.3f}'.format(simulation_result))
    print('result of exact solution: {:.3f}'.format(exact_solution))
    print('result of s_cum_mean: {:.3f}'.format(s_cum_mean[-1]))

    plt.figure(figsize=(10, 10))
    plt.suptitle('call option price by MC')
    sim_list = [100, 1000, 10000, 100000]

    for i in range(len(sim_list)):
        plt.subplot(2, 2, i + 1)
        plt.plot(s_cum_mean[:sim_list[i]], label='{} iteration'.format(sim_list[i]))
        plt.hlines(exact_solution, 0, sim_list[i], color='r')
        plt.legend()

    plt.show()

    # plt.plot(s_cum_mean, color='c')
    # plt.hlines(exact_solution, 0, nSimulation - 1, color='r')
    # plt.show()


if __name__ == '__main__':
    call_option_by_MC()

"""빨간선 = exact soluation (BS로 계산한)

-> 100회나 1000회는 exact value랑 차이가 있지만, 횟수가 증가할 수록 빨간선에 수렴하는 모습을 보인다

물론 횟수가 증가할 수록 계산 시간도 증가하니
MC를 사용할 때는 계산의 정확성과 계산 시간의 효율성을 비교해서 적절한 시뮬레이션 횟수를 결정해야 한다 (trade off 관계에 있음)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

S0 = 100.0     # 현재 주가
K = 100.0      # 행사가격
r = 0.05       # 무위험이자율
sigma = 0.20   # 변동성
T = 1.0        # 만기

def terminal_price(S0, r, sigma, T, Z):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

def terminal_price(S0, r, sigma, T, Z):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    # GBM 해: S(T) = S(0) · exp[(r − σ²/2)T + σ√T·Z]

def mc_european_call(S0, K, r, sigma, T, N, rng):
    Z = rng.standard_normal(N)                     # 난수 생성
    ST = terminal_price(S0, r, sigma, T, Z)        # 만기 주가
    payoff = np.maximum(ST - K, 0.0)               # payoff
    discounted = np.exp(-r * T) * payoff           # 할인 (각 표본에 먼저 적용)
    price = discounted.mean()                      # 평균 구하기
    se = discounted.std(ddof=1) / np.sqrt(N)       # 표준오차 = s / √N
    return price, se

# 유럽형 콜옵션

def bs_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# BS 해석해

def mc_asian_call(S0, K, r, sigma, T, N, M, rng):
    # 만기까지 M개 시점으로 나눠 경로 전체를 시뮬레이션하기
    dt = T / M
    Z = rng.standard_normal((N, M))                              # (경로 수, 시점 수)
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.cumsum(log_increments, axis=1))       # 각 시점의 주가
    avg_price = paths.mean(axis=1)                               # 기간 평균 A
    discounted = np.exp(-r * T) * np.maximum(avg_price - K, 0.0) # 바뀐 payoff
    return discounted.mean(), discounted.std(ddof=1) / np.sqrt(N)

# 아시안 콜옵션

def mc_european_call_antithetic(S0, K, r, sigma, T, N, rng):
    # 난수 총 N개 = Z (N/2개) + −Z (N/2개)
    Z = rng.standard_normal(N // 2)
    disc = np.exp(-r * T)
    payoff_plus = disc * np.maximum(terminal_price(S0, r, sigma, T, Z) - K, 0.0)
    payoff_minus = disc * np.maximum(terminal_price(S0, r, sigma, T, -Z) - K, 0.0)
    pair_avg = 0.5 * (payoff_plus + payoff_minus)   # 쌍 평균: 오차가 서로 상쇄
    return pair_avg.mean(), pair_avg.std(ddof=1) / np.sqrt(N // 2)

# 대조변수법

"""여기서부터 출력코드"""

# 유럽형 콜옵션

bs = bs_call(S0, K, r, sigma, T)
print(f"Black-Scholes 가격: {bs:.4f}\n")
print(f"{'N':>10} {'MC 가격':>9} {'SE':>8} {'95% 신뢰구간':>15} {'BS 포함':>10}")

rng = np.random.default_rng(2026)
Ns = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
results = []
for N in Ns:
    price, se = mc_european_call(S0, K, r, sigma, T, N, rng)
    lo, hi = price - 1.96 * se, price + 1.96 * se
    results.append((N, price, se))
    print(f"{N:>10,} {price:>11.4f} {se:>10.4f}   [{lo:>7.4f}, {hi:>7.4f}] {'O' if lo <= bs <= hi else 'X':>6}")

# 수렴 그래프 그리기

Ns_arr = np.array([n for n, _, _ in results])
prices = np.array([p for _, p, _ in results])
ses = np.array([s for _, _, s in results])

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(Ns_arr, prices, yerr=1.96 * ses, fmt="o", capsize=5, label="MC estimate (95% CI)")
ax.axhline(bs, color="tab:orange", linestyle="--", label=f"Black-Scholes = {bs:.4f}")
ax.set_xscale("log")
ax.set_xlabel("Number of simulations N")
ax.set_ylabel("Call price")
ax.set_title("Monte Carlo convergence to Black-Scholes")
ax.legend()
fig.tight_layout()

plt.show()

# 아시안 콜옵션

rng = np.random.default_rng(42)
price, se = mc_asian_call(S0, K, r, sigma, T, N=100_000, M=252, rng=rng)
print(f"아시안 콜 MC 가격: {price:.4f}  (SE {se:.4f})")
print(f"유럽형 콜 BS 가격: {bs:.4f}")
# 평균을 내면 변동성이 줄어서 더 싸다

N, M = 100_000, 252
dt = T / M
Z = rng.standard_normal((N, M))
paths = S0 * np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1))

S_T = paths[:, -1]            # 유럽형이 보는 값: 만기 가격
A = paths.mean(axis=1)        # 아시안이 보는 값: 기간 평균

disc = np.exp(-r * T)
print(f"아시안 콜 가격: {disc * np.maximum(A - K, 0).mean():.4f}")
print(f"표준편차  S(T): {S_T.std():.2f}   Ā: {A.std():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 경로와 누적 평균
ax = axes[0]
t = np.arange(1, M + 1)
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
for i, c in enumerate(colors):
    running_avg = np.cumsum(paths[i]) / t
    ax.plot(t, paths[i], color=c, lw=1.2, alpha=0.8)
    ax.plot(t, running_avg, color=c, lw=2.5, linestyle="--")
    ax.scatter(M, S_T[i], color=c, s=40, zorder=3)
ax.axhline(K, color="black", linestyle=":", lw=1.5)
ax.plot([], [], color="gray", lw=1.2, label="Price path")
ax.plot([], [], color="gray", lw=2.5, linestyle="--", label="Running average (ends at Ā)")
ax.plot([], [], color="black", linestyle=":", label="Strike K = 100")
ax.set_xlabel("Trading day")
ax.set_ylabel("Stock price")
ax.set_title("5 simulated paths and their running averages")
ax.legend(loc="upper left")

# 분포 비교
ax = axes[1]
bins = np.linspace(50, 180, 80)
ax.hist(S_T, bins=bins, alpha=0.5, density=True, label=f"S(T)  (std {S_T.std():.1f})")
ax.hist(A, bins=bins, alpha=0.5, density=True, label=f"Ā  (std {A.std():.1f})")
ax.axvline(K, color="black", linestyle=":", lw=1.5, label="Strike K = 100")
ax.set_xlabel("Price")
ax.set_ylabel("Density")
ax.set_title("Distribution: terminal price vs. average price")
ax.legend()

fig.tight_layout()
fig.savefig("asian_visualization.png", dpi=150)
plt.show()

"""실선 : 주가 / 점선 : 누적평균
-> 누적평균은 완만한 경향
"""

# 대조 변수법

p_plain, se_plain = mc_european_call(S0, K, r, sigma, T, 100_000, rng)
p_anti, se_anti = mc_european_call_antithetic(S0, K, r, sigma, T, 100_000, rng)
print(f"일반 MC    : {p_plain:.4f}  (SE {se_plain:.4f})")
print(f"대조변수법 : {p_anti:.4f}  (SE {se_anti:.4f})")
print(f"표준오차 감소: {1 - se_anti / se_plain:.1%},  분산 비율: {(se_anti / se_plain)**2:.2f}")