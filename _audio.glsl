#define LOOP_DURATION 30.
#define PITCH_SMOOTHING

#define TAU 6.283185

#define FORMANT_STEP .02
#define PHASE_STEP .04

#define FP(ftime,amp,f1,f2,f3){ if(ftimes.y<time){ ftimes=vec2(ftimes.y,ftime); fprev=fnext; fnext=vec4(f1,f2,f3,amp); } }

// After all the `PP`s are executed, `time` should be between `ptimes.y` and `ptimes.z`:
//
// --------------------------------------------------- TIME -->
//  ptimes.x       ptimes.y      time       ptimes.z       ptimes.w
//     |               |          .            |               |
//     |               |          .            |               |
//  phases.x       phases.y (interpolated   phases.z       phases.w
//     |               |        phase)         |               |
//     |               |          .            |               |

#define PP(ptime, phase) { if (ptimes.z < time) { ptimes = vec4(ptimes.yzw, ptime); phases = vec4(phases.yzw, phase); } }


float interpolate_phase(vec4 samples, float mu) {
    
    #ifdef PITCH_SMOOTHING
        // Catmull-Rom interpolation; continuous slope
        float a0, a1, a2, a3, mu2;
        mu2 = mu * mu;

        a0 = -0.5*samples.x + 1.5*samples.y - 1.5*samples.z + 0.5*samples.w;
        a1 = samples.x - 2.5*samples.y + 2.*samples.z - 0.5*samples.w;
        a2 = -0.5*samples.x + 0.5*samples.z;
        a3 = samples.y;

        return a0*mu*mu2 + a1*mu2 + a2*mu + a3;
    #else
        // Linear interpolation; discontinuous slope
        return samples.y*(1.-mu) + samples.z*mu;
    #endif
}

float bandpass(float freq, float formant) {
    float q = 16000.;
    float d = freq - formant;
    return q / (d * d + q);
}

float pulse(float time, float pulse_time, float pulse_length) {
    float d = (time - pulse_time) / pulse_length;
    return .01 / (pow(d, 4.) + .01);
}

#define ITERATIONS 8
#define MOD2 vec2(443.8975,397.2973)
#define MOD3 vec3(443.8975,397.2973, 491.1871)
// Hash function by David Hoskins
vec2 hash22(vec2 p)
{
	vec3 p3 = fract(vec3(p.xyx) * MOD3);
    p3 += dot(p3.zxy, p3.yxz+19.19);
    return fract(vec2(p3.x * p3.y, p3.z*p3.x));
}
float hash22mono(float p) {
    vec2 stereo = hash22(vec2(p, p + 10.)) * .5;
    return stereo.x + stereo.y;
}

vec2 mainSound(float time)
{
    time = mix(time, time - LOOP_DURATION, step(LOOP_DURATION, time));

    vec2 ftimes = vec2(0.);
    // vec4: F1, F2, F3, amplitude
    vec4 fnext = vec4(0.), fprev = vec4(0.);
    
    
FP(0.00,0.000020,339.,1464.,2554.)FP(0.06,0.000400,353.,1420.,2552.)FP(0.08,0.002000,400.,792.,2551.)FP(0.14,0.018000,821.,1166.,2947.)FP(0.18,0.022000,839.,1059.,2649.)FP(0.20,0.022000,935.,1526.,2622.)FP(0.22,0.036000,961.,1646.,2495.)FP(0.24,0.072000,755.,1796.,2595.)FP(0.30,0.018000,519.,2050.,2843.)FP(0.38,0.004000,420.,1779.,2681.)FP(0.44,0.004000,321.,2203.,2836.)FP(0.54,0.001000,308.,2435.,2769.)FP(0.56,0.000500,642.,2499.,3524.)FP(0.58,0.000200,128.,2621.,3593.)FP(0.60,0.000200,1089.,2537.,3763.)FP(0.62,0.000200,1456.,2465.,3848.)FP(0.64,0.000800,272.,2432.,3643.)FP(0.66,0.006000,412.,1431.,2384.)FP(0.72,0.019000,739.,1492.,2412.)FP(0.82,0.007000,396.,1287.,2011.)FP(0.86,0.000900,369.,1043.,2093.)FP(0.88,0.000300,390.,1253.,2087.)FP(0.90,0.000600,389.,630.,1392.)FP(0.92,0.000800,402.,634.,1401.)FP(0.94,0.010000,415.,734.,3096.)FP(0.96,0.010000,443.,856.,3174.)FP(0.98,0.008000,850.,1221.,2972.)FP(1.02,0.025000,1219.,1619.,3008.)FP(1.06,0.044000,1109.,1590.,2934.)FP(1.08,0.044000,1279.,1721.,2887.)FP(1.10,0.154000,952.,1590.,2933.)FP(1.14,0.107000,945.,1434.,3595.)FP(1.16,0.082000,672.,1471.,3677.)FP(1.20,0.039000,579.,1560.,3604.)FP(1.22,0.002000,432.,1605.,2794.)FP(1.28,0.000800,427.,2111.,2711.)FP(1.30,0.001000,425.,2156.,3479.)FP(1.36,0.000200,337.,1686.,3363.)FP(1.44,0.002000,442.,1481.,3575.)FP(1.46,0.020000,596.,1538.,2495.)FP(1.58,0.000900,544.,1438.,2488.)FP(1.64,0.000020,665.,1525.,2736.)FP(1.66,0.000600,632.,1495.,2440.)FP(1.68,0.000800,325.,1354.,1992.)FP(1.70,0.003000,263.,1248.,2014.)FP(1.74,0.115000,353.,1922.,2437.)FP(1.78,0.071000,271.,2108.,2515.)FP(1.84,0.016000,264.,2210.,2696.)FP(1.86,0.002000,243.,2055.,4025.)FP(1.88,0.001000,264.,2149.,2668.)FP(1.90,0.001000,289.,2040.,2709.)FP(1.92,0.001000,191.,2073.,2637.)FP(1.94,0.001000,164.,2436.,2987.)FP(1.96,0.000030,221.,2766.,3864.)FP(1.98,0.000010,239.,1556.,3637.)FP(2.00,0.001000,232.,1112.,3200.)FP(2.02,0.007000,993.,1602.,3029.)FP(2.04,0.043000,382.,1660.,3223.)FP(2.06,0.047000,540.,1573.,2903.)FP(2.08,0.097000,982.,1560.,3267.)FP(2.10,0.106000,985.,1759.,2608.)FP(2.12,0.106000,826.,1216.,2944.)FP(2.14,0.051000,830.,1196.,2791.)FP(2.20,0.061000,874.,1233.,3092.)FP(2.24,0.068000,923.,1392.,2671.)FP(2.28,0.074000,617.,1507.,2455.)FP(2.30,0.029000,883.,1420.,2323.)FP(2.32,0.002000,504.,2089.,3354.)FP(2.34,0.001000,872.,1510.,2535.)FP(2.36,0.000200,919.,1353.,2358.)FP(2.38,0.000050,610.,1573.,2713.)FP(2.40,0.000900,890.,1550.,2643.)FP(2.42,0.002000,833.,2347.,3795.)FP(2.44,0.004000,339.,1916.,2551.)FP(2.46,0.005000,290.,1811.,2491.)FP(2.48,0.008000,328.,1582.,2387.)FP(2.50,0.008000,447.,1479.,2399.)FP(2.58,0.002000,431.,1278.,2390.)FP(2.60,0.000400,256.,1260.,2400.)FP(2.62,0.000400,249.,2423.,3341.)FP(2.64,0.005000,253.,2432.,3309.)FP(2.66,0.007000,366.,2519.,3390.)FP(2.68,0.012000,396.,2060.,2630.)FP(2.76,0.013000,417.,2115.,2570.)FP(2.78,0.005000,394.,2251.,3366.)FP(2.80,0.000600,246.,2143.,2789.)FP(2.82,0.000030,335.,1220.,2126.)FP(2.84,0.000009,196.,1294.,2201.)FP(2.86,0.000020,1063.,2445.,3226.)FP(2.88,0.000200,1402.,3282.,3454.)FP(2.90,0.000200,237.,1457.,3298.)FP(2.92,0.000600,247.,1248.,3450.)FP(2.94,0.005000,293.,1157.,2519.)FP(2.96,0.007000,343.,1258.,2507.)FP(2.98,0.013000,305.,1341.,2454.)FP(3.02,0.154000,492.,2093.,2659.)FP(3.10,0.219000,389.,2292.,2968.)FP(3.22,0.213000,567.,2131.,2848.)FP(3.24,0.212000,471.,2128.,2734.)FP(3.34,0.134000,512.,2160.,2577.)FP(3.46,0.012000,387.,2253.,2566.)FP(3.54,0.000100,347.,1779.,2456.)FP(3.58,0.000200,244.,1509.,2355.)FP(3.60,0.000200,294.,1626.,2329.)FP(3.62,0.000080,745.,1667.,2499.)FP(3.64,0.000070,1460.,2436.,3921.)FP(3.66,0.000070,178.,2381.,4027.)FP(3.68,0.000060,221.,1490.,4212.)FP(3.70,0.000030,103.,2371.,4079.)FP(3.72,0.000010,97.,1769.,3000.)FP(3.74,0.000000,173.,2817.,3727.)FP(3.76,0.000000,500.,1500.,2500.)FP(4.40,0.000000,500.,1500.,2500.)FP(4.42,0.000000,104.,2081.,2718.)FP(4.44,0.000001,246.,2087.,2751.)FP(4.46,0.000070,180.,1337.,2625.)FP(4.48,0.002000,307.,1945.,2619.)FP(4.50,0.007000,327.,1933.,2509.)FP(4.56,0.020000,602.,1328.,2320.)FP(4.60,0.016000,611.,910.,2370.)FP(4.62,0.000600,671.,1281.,2356.)FP(4.68,0.000030,666.,1344.,2247.)FP(4.70,0.000300,973.,2543.,3246.)FP(4.72,0.002000,1033.,2450.,3529.)FP(4.74,0.016000,978.,2366.,2754.)FP(4.76,0.067000,1053.,2055.,2906.)FP(4.78,0.067000,780.,1221.,2265.)FP(4.80,0.064000,595.,1275.,2263.)FP(4.86,0.051000,528.,1232.,1599.)FP(4.88,0.010000,383.,1580.,2140.)FP(4.90,0.046000,516.,1147.,2321.)FP(4.94,0.052000,514.,1026.,2619.)FP(4.96,0.018000,455.,988.,2718.)FP(4.98,0.004000,285.,1009.,2733.)FP(5.02,0.001000,238.,993.,2632.)FP(5.04,0.002000,238.,1037.,2675.)FP(5.06,0.013000,307.,937.,2595.)FP(5.08,0.029000,702.,959.,3357.)FP(5.10,0.029000,826.,1100.,2826.)FP(5.16,0.036000,845.,1759.,2615.)FP(5.18,0.020000,678.,1798.,2625.)FP(5.20,0.015000,715.,1773.,2690.)FP(5.22,0.003000,368.,1773.,2731.)FP(5.26,0.007000,589.,1287.,2720.)FP(5.30,0.027000,861.,1308.,2774.)FP(5.34,0.016000,993.,1357.,2710.)FP(5.36,0.018000,708.,1108.,2815.)FP(5.40,0.010000,859.,1223.,2978.)FP(5.42,0.011000,1060.,1361.,3071.)FP(5.46,0.008000,1112.,1894.,2835.)FP(5.48,0.007000,788.,1969.,2859.)FP(5.50,0.010000,793.,2003.,2724.)FP(5.52,0.010000,573.,2015.,2680.)FP(5.56,0.002000,412.,2246.,2874.)FP(5.58,0.000300,242.,2233.,2823.)FP(5.60,0.000070,233.,950.,2315.)FP(5.62,0.000020,511.,1799.,3170.)FP(5.64,0.000080,542.,1475.,2983.)FP(5.66,0.000100,686.,2956.,3880.)FP(5.68,0.000200,261.,1965.,3839.)FP(5.74,0.002000,387.,1911.,2322.)FP(5.80,0.007000,436.,1464.,2408.)FP(5.82,0.008000,551.,1242.,2338.)FP(5.84,0.008000,560.,1351.,2429.)FP(5.86,0.012000,735.,1298.,2415.)FP(5.92,0.005000,735.,1575.,2556.)FP(5.94,0.002000,678.,1399.,1805.)FP(5.98,0.005000,586.,1479.,2085.)FP(6.00,0.013000,566.,2085.,2797.)FP(6.04,0.013000,515.,2107.,2811.)FP(6.12,0.024000,583.,1692.,2512.)FP(6.14,0.016000,547.,1305.,2461.)FP(6.16,0.001000,531.,1250.,1825.)FP(6.18,0.000500,409.,1130.,2047.)FP(6.20,0.000400,401.,1105.,1997.)FP(6.22,0.001000,373.,1122.,1917.)FP(6.24,0.004000,350.,1464.,1867.)FP(6.26,0.004000,388.,1861.,2298.)FP(6.28,0.004000,518.,1937.,2568.)FP(6.32,0.003000,398.,2052.,2819.)FP(6.34,0.003000,436.,1935.,2805.)FP(6.36,0.000200,239.,2092.,2820.)FP(6.38,0.000030,259.,2110.,2851.)FP(6.42,0.000004,506.,1963.,2521.)FP(6.44,0.000005,519.,1913.,2664.)FP(6.46,0.000040,511.,2030.,2794.)FP(6.48,0.000200,275.,1961.,2872.)FP(6.50,0.003000,395.,2047.,2682.)FP(6.62,0.014000,428.,2174.,2640.)FP(6.72,0.000100,322.,2307.,2383.)FP(6.74,0.000040,288.,1190.,2664.)FP(6.76,0.000000,277.,1226.,2168.)FP(6.78,0.000000,324.,1844.,2685.)FP(6.80,0.000000,442.,1374.,2417.)FP(6.82,0.000001,858.,1289.,2403.)FP(6.84,0.000030,698.,1358.,2458.)FP(6.90,0.000900,856.,1427.,2508.)FP(6.96,0.007000,390.,1630.,2390.)FP(6.98,0.000100,553.,1627.,2320.)FP(7.00,0.000030,512.,1316.,2284.)FP(7.02,0.003000,527.,1476.,2851.)FP(7.04,0.011000,1040.,1431.,3090.)FP(7.06,0.011000,931.,1398.,2688.)FP(7.08,0.004000,323.,1108.,2503.)FP(7.10,0.005000,521.,1236.,2526.)FP(7.14,0.000900,252.,1273.,2529.)FP(7.20,0.006000,347.,1392.,2846.)FP(7.22,0.046000,644.,1564.,2809.)FP(7.28,0.078000,941.,1474.,2677.)FP(7.34,0.015000,517.,1785.,2647.)FP(7.38,0.003000,343.,1537.,2525.)FP(7.42,0.000800,300.,1359.,2526.)FP(7.44,0.010000,933.,1255.,2651.)FP(7.46,0.013000,460.,1074.,2583.)FP(7.50,0.013000,449.,852.,2877.)FP(7.64,0.030000,713.,1018.,2572.)FP(7.70,0.010000,522.,1032.,2849.)FP(7.72,0.001000,311.,1086.,2843.)FP(7.74,0.002000,273.,1084.,2705.)FP(7.76,0.003000,376.,998.,2703.)FP(7.82,0.005000,406.,842.,2719.)FP(7.84,0.006000,414.,986.,2627.)FP(7.88,0.008000,392.,1374.,2501.)FP(7.92,0.000600,291.,1411.,2448.)FP(7.96,0.000600,443.,1427.,2388.)FP(7.98,0.010000,335.,1399.,2398.)FP(8.00,0.057000,594.,1445.,2511.)FP(8.04,0.066000,842.,1415.,2659.)FP(8.10,0.036000,922.,1399.,2672.)FP(8.14,0.024000,758.,1305.,2630.)FP(8.20,0.012000,524.,1616.,2262.)FP(8.22,0.000400,697.,1297.,3315.)FP(8.24,0.000200,505.,1253.,2942.)FP(8.28,0.000020,457.,1596.,2487.)FP(8.30,0.000020,892.,2079.,2618.)FP(8.32,0.000300,1413.,2045.,2626.)FP(8.34,0.001000,314.,1802.,2326.)FP(8.36,0.002000,376.,1789.,2488.)FP(8.38,0.002000,361.,1222.,2570.)FP(8.42,0.000900,337.,1058.,2585.)FP(8.48,0.000800,302.,1187.,2572.)FP(8.52,0.006000,341.,1892.,2537.)FP(8.58,0.003000,297.,2173.,2734.)FP(8.62,0.001000,309.,2586.,3478.)FP(8.64,0.000400,229.,2514.,3950.)FP(8.66,0.000100,483.,3043.,3896.)FP(8.68,0.000070,496.,2721.,3448.)FP(8.70,0.000070,469.,3044.,4004.)FP(8.72,0.000200,485.,1266.,3920.)FP(8.74,0.004000,409.,1491.,2591.)FP(8.82,0.019000,659.,1342.,2470.)FP(8.86,0.010000,664.,1242.,2575.)FP(8.90,0.001000,369.,1407.,2594.)FP(8.92,0.000100,704.,1336.,2596.)FP(8.94,0.000030,554.,1203.,2371.)FP(9.00,0.000002,535.,1442.,2734.)FP(9.02,0.000001,532.,1201.,2424.)FP(9.04,0.000000,500.,1500.,2500.)FP(9.76,0.000000,500.,1500.,2500.)FP(9.78,0.000000,204.,2252.,3247.)FP(9.80,0.000000,98.,1957.,2609.)FP(9.82,0.000001,284.,1469.,2182.)FP(9.84,0.000001,226.,1222.,2134.)FP(9.86,0.000001,241.,1193.,2192.)FP(9.88,0.000002,199.,1032.,2018.)FP(9.90,0.000004,219.,1147.,1988.)FP(9.92,0.000006,179.,1076.,1906.)FP(9.94,0.000009,146.,1147.,2503.)FP(9.96,0.000030,186.,1947.,2701.)FP(9.98,0.000090,142.,2362.,2808.)FP(10.00,0.001000,221.,2421.,3028.)FP(10.02,0.011000,378.,2519.,2890.)FP(10.06,0.011000,365.,2255.,2736.)FP(10.08,0.015000,255.,2680.,3650.)FP(10.10,0.022000,824.,2690.,3573.)FP(10.12,0.022000,421.,2745.,4028.)FP(10.14,0.005000,336.,2646.,4048.)FP(10.16,0.006000,172.,2575.,4053.)FP(10.18,0.010000,1476.,2428.,3412.)FP(10.20,0.010000,593.,2623.,3998.)FP(10.22,0.002000,319.,2517.,4144.)FP(10.24,0.046000,266.,1058.,2630.)FP(10.26,0.046000,513.,1162.,2627.)FP(10.28,0.008000,322.,794.,2552.)FP(10.32,0.011000,328.,1012.,2356.)FP(10.34,0.008000,305.,790.,2417.)FP(10.36,0.008000,233.,842.,2478.)FP(10.42,0.005000,275.,817.,2048.)FP(10.46,0.005000,287.,861.,2620.)FP(10.50,0.005000,235.,871.,2665.)FP(10.52,0.005000,231.,932.,2249.)FP(10.54,0.005000,210.,937.,2504.)FP(10.56,0.005000,262.,950.,2359.)FP(10.60,0.002000,266.,860.,2401.)FP(10.62,0.005000,290.,2209.,2516.)FP(10.68,0.022000,291.,2399.,2847.)FP(10.70,0.026000,223.,2398.,2909.)FP(10.74,0.083000,287.,2642.,3026.)FP(10.76,0.059000,957.,2681.,3052.)FP(10.84,0.006000,866.,2721.,3250.)FP(10.86,0.007000,1291.,2470.,3203.)FP(10.88,0.007000,452.,1523.,2486.)FP(10.90,0.000600,434.,1650.,3186.)FP(10.92,0.000600,594.,1605.,2448.)FP(10.94,0.000010,448.,1179.,2604.)FP(10.96,0.000003,186.,1105.,2746.)FP(10.98,0.000001,553.,1549.,2343.)FP(11.00,0.000001,543.,1253.,2387.)FP(11.04,0.000000,535.,1586.,2211.)FP(11.06,0.000000,500.,1500.,2500.)FP(12.20,0.000000,500.,1500.,2500.)FP(12.22,0.000100,773.,2588.,3579.)FP(12.24,0.001000,393.,1886.,2865.)FP(12.26,0.002000,217.,1943.,2723.)FP(12.30,0.007000,302.,2008.,2372.)FP(12.32,0.007000,380.,1657.,2079.)FP(12.36,0.015000,543.,1132.,2340.)FP(12.38,0.015000,678.,1141.,2294.)FP(12.48,0.010000,612.,1088.,1660.)FP(12.52,0.014000,652.,1023.,2457.)FP(12.56,0.011000,524.,928.,2835.)FP(12.58,0.008000,488.,929.,2840.)FP(12.60,0.009000,489.,938.,3650.)FP(12.62,0.024000,527.,935.,2853.)FP(12.66,0.027000,943.,1171.,2887.)FP(12.72,0.069000,840.,1103.,2924.)FP(12.80,0.070000,887.,1361.,2947.)FP(12.82,0.101000,965.,1781.,2781.)FP(12.88,0.078000,801.,2171.,2807.)FP(12.90,0.058000,887.,2154.,2781.)FP(12.92,0.058000,382.,2315.,2848.)FP(12.96,0.010000,328.,2390.,2667.)FP(13.00,0.014000,480.,2130.,2702.)FP(13.02,0.028000,728.,1854.,2613.)FP(13.06,0.102000,941.,1524.,2362.)FP(13.12,0.057000,670.,1843.,2413.)FP(13.14,0.013000,795.,1747.,2564.)FP(13.16,0.003000,242.,2003.,2487.)FP(13.18,0.019000,245.,1810.,2457.)FP(13.20,0.056000,405.,2003.,2594.)FP(13.28,0.086000,665.,2032.,2659.)FP(13.32,0.038000,561.,1357.,2543.)FP(13.44,0.090000,482.,2146.,2790.)FP(13.56,0.010000,345.,2152.,2830.)FP(13.66,0.054000,564.,2072.,2796.)FP(13.68,0.049000,382.,2166.,2708.)FP(13.70,0.049000,334.,2289.,2838.)FP(13.72,0.005000,443.,2158.,2784.)FP(13.74,0.000900,330.,2236.,2942.)FP(13.76,0.007000,314.,2997.,3828.)FP(13.80,0.019000,503.,2838.,3660.)FP(13.82,0.019000,312.,2822.,3436.)FP(13.86,0.014000,263.,2600.,3376.)FP(13.90,0.027000,277.,2115.,3109.)FP(13.92,0.027000,323.,2133.,3256.)FP(13.94,0.028000,466.,1902.,2379.)FP(14.16,0.000050,424.,1119.,2223.)FP(14.18,0.000020,467.,1505.,2222.)FP(14.20,0.000002,440.,2275.,3293.)FP(14.24,0.000000,584.,2306.,3234.)FP(14.26,0.000000,542.,1623.,2294.)FP(14.28,0.000000,621.,1642.,3081.)FP(14.30,0.000000,500.,1500.,2500.)FP(14.32,0.000000,597.,1740.,2350.)FP(14.34,0.000000,503.,1887.,2496.)FP(14.36,0.000000,754.,1988.,2582.)FP(14.38,0.000000,675.,1750.,2314.)FP(14.40,0.000001,351.,1748.,2313.)FP(14.42,0.000001,702.,1880.,2334.)FP(14.44,0.000000,719.,1933.,2347.)FP(14.46,0.000000,683.,1822.,2342.)FP(14.48,0.000000,298.,2307.,2981.)FP(14.50,0.000004,756.,1610.,2535.)FP(14.52,0.001000,308.,1675.,2982.)FP(14.60,0.074000,284.,2100.,3194.)FP(14.68,0.088000,796.,2367.,3333.)FP(14.70,0.088000,836.,1997.,2562.)FP(14.72,0.073000,859.,1495.,2627.)FP(14.74,0.073000,871.,1407.,2737.)FP(14.76,0.073000,876.,1437.,2268.)FP(14.80,0.073000,898.,1630.,2227.)FP(14.82,0.030000,686.,1817.,2838.)FP(14.84,0.006000,608.,1562.,2489.)FP(14.86,0.003000,605.,1821.,2871.)FP(14.88,0.001000,695.,1732.,2750.)FP(14.90,0.000600,619.,1786.,2751.)FP(14.92,0.000200,882.,1914.,2782.)FP(14.94,0.000100,563.,1765.,3513.)FP(14.96,0.000600,815.,1624.,3028.)FP(14.98,0.004000,846.,1674.,3497.)FP(15.00,0.013000,912.,1779.,3039.)FP(15.02,0.091000,652.,2058.,2948.)FP(15.04,0.091000,821.,1943.,3147.)FP(15.08,0.044000,767.,1795.,2348.)FP(15.34,0.098000,753.,1863.,2583.)FP(15.40,0.067000,542.,1993.,2453.)FP(15.42,0.021000,571.,1915.,2589.)FP(15.44,0.016000,437.,2107.,2603.)FP(15.48,0.157000,540.,2134.,2571.)FP(15.52,0.052000,459.,2189.,2865.)FP(15.56,0.071000,501.,2191.,2550.)FP(15.64,0.104000,775.,1810.,2214.)FP(15.66,0.019000,771.,1841.,2935.)FP(15.68,0.003000,701.,2188.,2952.)FP(15.70,0.003000,1061.,1964.,2524.)FP(15.72,0.001000,618.,1958.,2551.)FP(15.74,0.000500,481.,1617.,2199.)FP(15.76,0.000100,670.,1816.,2895.)FP(15.78,0.002000,889.,1987.,2936.)FP(15.80,0.004000,359.,1014.,3371.)FP(15.82,0.011000,288.,1208.,3408.)FP(15.84,0.024000,815.,1215.,3381.)FP(15.86,0.056000,510.,1245.,3362.)FP(15.88,0.056000,680.,1394.,3278.)FP(15.90,0.034000,748.,1194.,2374.)FP(16.04,0.048000,874.,1215.,2291.)FP(16.10,0.065000,735.,1415.,2335.)FP(16.14,0.048000,633.,1288.,2465.)FP(16.20,0.012000,736.,1490.,2447.)FP(16.24,0.019000,374.,1199.,3224.)FP(16.26,0.032000,368.,1544.,3216.)FP(16.28,0.140000,445.,1993.,2602.)FP(16.30,0.140000,621.,1974.,2553.)FP(16.34,0.072000,343.,2249.,2667.)FP(16.46,0.006000,235.,2191.,2761.)FP(16.50,0.003000,329.,2175.,2730.)FP(16.54,0.086000,418.,1419.,2665.)FP(16.58,0.118000,849.,1251.,2513.)FP(16.62,0.044000,853.,1323.,2427.)FP(16.64,0.025000,843.,1293.,3053.)FP(16.68,0.044000,803.,1327.,2798.)FP(16.72,0.047000,963.,1226.,3025.)FP(16.82,0.031000,923.,1259.,3066.)FP(16.84,0.031000,882.,1202.,2642.)FP(16.98,0.000003,798.,1373.,2927.)FP(17.02,0.000001,394.,1053.,3106.)FP(17.04,0.000001,347.,856.,3043.)FP(17.08,0.000002,273.,851.,2289.)FP(17.12,0.000005,244.,779.,3550.)FP(17.14,0.000005,283.,804.,2929.)FP(17.16,0.000020,242.,806.,1784.)FP(17.18,0.000040,239.,981.,3118.)FP(17.20,0.000700,269.,1374.,3050.)FP(17.22,0.017000,533.,1092.,3098.)FP(17.26,0.040000,665.,1224.,2896.)FP(17.28,0.067000,889.,1266.,2771.)FP(17.40,0.052000,946.,1987.,2532.)FP(17.42,0.172000,926.,2013.,2606.)FP(17.44,0.173000,640.,1972.,2610.)FP(17.48,0.178000,489.,2773.,3137.)FP(17.54,0.054000,526.,2417.,3184.)FP(17.58,0.011000,486.,2142.,4178.)FP(17.60,0.011000,502.,2245.,3570.)FP(17.62,0.011000,579.,2121.,4286.)FP(17.64,0.011000,568.,2143.,3924.)FP(17.68,0.005000,757.,1967.,3727.)FP(17.70,0.002000,536.,1727.,3889.)FP(17.72,0.008000,410.,2160.,3704.)FP(17.74,0.028000,458.,2177.,3740.)FP(17.76,0.079000,707.,1836.,3895.)FP(17.84,0.022000,782.,1202.,2720.)FP(17.88,0.011000,801.,1367.,3267.)FP(17.90,0.008000,457.,805.,3196.)FP(17.92,0.006000,530.,849.,3103.)FP(17.94,0.003000,793.,1318.,3029.)FP(17.98,0.006000,781.,2277.,2924.)FP(18.02,0.022000,870.,2071.,3834.)FP(18.08,0.000900,619.,2081.,3987.)FP(18.10,0.000900,725.,1829.,2406.)FP(18.12,0.000500,651.,2251.,3682.)FP(18.16,0.000700,389.,2011.,3706.)FP(18.18,0.000700,671.,2137.,3881.)FP(18.20,0.000800,131.,2286.,3726.)FP(18.22,0.000800,189.,2122.,3637.)FP(18.24,0.000300,442.,1695.,3695.)FP(18.26,0.002000,558.,1559.,2131.)FP(18.28,0.096000,447.,2308.,3658.)FP(18.30,0.466000,587.,2311.,3874.)FP(18.40,0.080000,742.,1558.,1874.)FP(18.48,0.051000,649.,1839.,2189.)FP(18.58,0.000700,711.,1789.,2216.)FP(18.60,0.000700,626.,1458.,2009.)FP(18.62,0.004000,609.,1082.,2852.)FP(18.78,0.000200,578.,866.,3414.)FP(18.80,0.000060,565.,2229.,3193.)FP(18.82,0.000020,383.,770.,2985.)FP(18.84,0.000009,762.,2146.,3137.)FP(18.86,0.000010,869.,2599.,3149.)FP(18.90,0.000040,770.,1986.,3672.)FP(18.92,0.000040,847.,2046.,3722.)FP(18.94,0.009000,884.,2239.,2863.)FP(18.96,0.012000,353.,2235.,3715.)FP(18.98,0.016000,235.,2223.,3884.)FP(19.00,0.016000,252.,1865.,2252.)FP(19.08,0.014000,301.,2286.,2615.)FP(19.12,0.072000,325.,2105.,2477.)FP(19.14,0.072000,357.,2025.,2495.)FP(19.16,0.031000,447.,1359.,2496.)FP(19.18,0.031000,549.,1241.,2487.)FP(19.24,0.033000,877.,1234.,2367.)FP(19.26,0.035000,878.,1121.,2734.)FP(19.28,0.035000,877.,1826.,2704.)FP(19.32,0.032000,888.,1882.,2715.)FP(19.34,0.010000,351.,1869.,2637.)FP(19.36,0.002000,407.,1982.,2798.)FP(19.38,0.002000,335.,1826.,2737.)FP(19.40,0.005000,529.,2135.,2863.)FP(19.42,0.019000,579.,2259.,2801.)FP(19.48,0.020000,583.,1570.,2411.)FP(19.50,0.030000,559.,1630.,3936.)FP(19.54,0.023000,372.,1835.,3938.)FP(19.56,0.023000,494.,2682.,3759.)FP(19.58,0.016000,395.,2467.,3820.)FP(19.60,0.008000,454.,1686.,3777.)FP(19.64,0.009000,747.,2606.,3855.)FP(19.66,0.038000,496.,1677.,2502.)FP(19.70,0.077000,593.,1767.,2578.)FP(19.72,0.032000,819.,1759.,2624.)FP(19.74,0.032000,845.,2001.,2778.)FP(19.78,0.015000,829.,1439.,2828.)FP(19.80,0.016000,825.,1451.,2830.)FP(19.82,0.016000,702.,915.,2760.)FP(19.88,0.006000,506.,1035.,2821.)FP(19.90,0.023000,522.,1042.,2540.)FP(19.92,0.023000,771.,1182.,2821.)FP(20.10,0.074000,911.,1266.,2842.)FP(20.16,0.071000,789.,1711.,2429.)FP(20.18,0.033000,625.,1990.,2541.)FP(20.26,0.003000,462.,1920.,2520.)FP(20.28,0.000400,625.,1984.,2473.)FP(20.36,0.000030,629.,2117.,2481.)FP(20.38,0.000009,1004.,2348.,3124.)FP(20.40,0.000002,664.,2373.,3085.)FP(20.42,0.000002,640.,1904.,3168.)FP(20.44,0.000000,500.,1500.,2500.)FP(21.62,0.000000,500.,1500.,2500.)FP(21.64,0.000001,1026.,1762.,2685.)FP(21.66,0.000001,1056.,1916.,2398.)FP(21.70,0.004000,1052.,1842.,2704.)FP(21.72,0.011000,940.,1998.,3490.)FP(21.74,0.011000,311.,1795.,2351.)FP(21.80,0.007000,349.,1423.,2275.)FP(21.82,0.005000,386.,1589.,2330.)FP(21.88,0.010000,260.,2175.,2597.)FP(21.90,0.009000,264.,2304.,2862.)FP(21.98,0.009000,265.,2109.,2521.)FP(22.04,0.014000,327.,2018.,2406.)FP(22.06,0.014000,420.,1779.,2402.)FP(22.08,0.011000,366.,1727.,2476.)FP(22.12,0.004000,513.,1288.,2538.)FP(22.14,0.006000,550.,1300.,3248.)FP(22.16,0.006000,617.,1293.,2709.)FP(22.18,0.006000,625.,1244.,2848.)FP(22.20,0.004000,572.,1045.,2828.)FP(22.26,0.000200,385.,1052.,2989.)FP(22.28,0.000020,318.,833.,1904.)FP(22.30,0.000001,423.,1096.,2055.)FP(22.32,0.000000,310.,1179.,2035.)FP(22.34,0.000000,1212.,1908.,2870.) 

vec4 phases = vec4(0.);
    vec4 ptimes = vec4(0.);
    
PP(0.00,0.00)PP(0.04,7.10)PP(0.08,15.67)PP(0.12,24.99)PP(0.28,66.86)PP(0.32,76.77)PP(0.40,92.70)PP(0.44,100.49)PP(0.52,114.00)PP(0.64,137.53)PP(0.68,145.32)PP(0.72,152.73)PP(0.84,174.68)PP(0.92,190.67)PP(0.96,198.42)PP(1.12,235.93)PP(1.32,276.24)PP(1.56,323.47)PP(1.60,331.93)PP(1.68,350.15)PP(1.76,370.75)PP(2.04,436.65)PP(2.12,458.78)PP(2.24,496.25)PP(2.40,542.14)PP(2.56,580.87)PP(2.60,589.80)PP(2.76,623.58)PP(2.92,661.75)PP(3.00,685.74)PP(3.04,698.56)PP(3.20,755.46)PP(3.28,782.31)PP(3.44,816.76)PP(3.48,824.71)PP(4.44,1009.59)PP(4.48,1017.39)PP(4.52,1026.03)PP(4.68,1063.32)PP(4.80,1094.87)PP(4.92,1125.72)PP(5.00,1143.56)PP(5.08,1160.88)PP(5.32,1214.71)PP(5.44,1247.47)PP(5.48,1258.53)PP(5.52,1268.64)PP(5.56,1277.05)PP(5.68,1300.33)PP(5.72,1308.38)PP(5.76,1316.44)PP(5.84,1331.74)PP(5.88,1338.95)PP(5.92,1347.02)PP(6.00,1365.11)PP(6.08,1386.36)PP(6.12,1397.26)PP(6.16,1407.82)PP(6.24,1429.84)PP(6.28,1440.49)PP(6.32,1450.14)PP(6.36,1457.75)PP(6.48,1481.80)PP(6.56,1495.76)PP(6.60,1502.42)PP(6.68,1515.46)PP(6.72,1522.53)PP(7.00,1586.45)PP(7.08,1606.34)PP(7.12,1615.80)PP(7.16,1623.79)PP(7.20,1633.30)PP(7.24,1642.66)PP(7.32,1662.33)PP(7.36,1671.57)PP(7.40,1679.48)PP(7.44,1688.20)PP(7.52,1705.82)PP(7.64,1736.61)PP(7.68,1746.94)PP(7.72,1756.90)PP(7.76,1765.81)PP(7.80,1774.05)PP(7.88,1788.93)PP(7.92,1795.54)PP(7.96,1802.51)PP(8.00,1810.87)PP(8.08,1829.29)PP(8.16,1850.68)PP(8.20,1860.92)PP(8.24,1870.24)PP(8.32,1886.80)PP(8.40,1899.69)PP(8.44,1905.63)PP(8.48,1911.45)PP(8.56,1923.43)PP(8.72,1951.78)PP(8.80,1967.64)PP(8.84,1976.67)PP(8.88,1986.00)PP(9.96,2203.22)PP(10.04,2218.47)PP(10.28,2261.62)PP(10.36,2274.94)PP(12.20,2582.81)PP(12.24,2589.57)PP(12.28,2596.98)PP(12.36,2615.52)PP(12.44,2634.66)PP(12.52,2653.24)PP(12.60,2672.44)PP(12.64,2682.26)PP(12.72,2703.82)PP(12.80,2725.97)PP(12.88,2747.39)PP(12.96,2766.13)PP(13.00,2774.41)PP(13.04,2782.35)PP(13.08,2790.90)PP(13.16,2808.68)PP(13.20,2818.50)PP(13.28,2840.68)PP(13.32,2851.76)PP(13.36,2862.43)PP(13.40,2872.39)PP(13.48,2888.36)PP(13.56,2906.48)PP(13.64,2929.38)PP(13.68,2941.26)PP(13.84,2983.01)PP(13.96,3009.60)PP(14.52,3128.13)PP(14.56,3136.71)PP(14.60,3146.61)PP(14.68,3169.33)PP(14.80,3206.59)PP(14.88,3231.14)PP(15.00,3272.11)PP(15.08,3302.68)PP(15.20,3348.66)PP(15.44,3436.24)PP(15.64,3509.30)PP(15.84,3584.08)PP(15.92,3615.78)PP(16.08,3676.23)PP(16.12,3690.76)PP(16.16,3705.83)PP(16.20,3720.51)PP(16.24,3733.10)PP(16.32,3758.09)PP(16.40,3780.57)PP(16.56,3825.75)PP(16.64,3847.31)PP(16.72,3866.20)PP(16.76,3874.50)PP(16.80,3881.29)PP(16.84,3886.85)PP(16.88,3893.03)PP(17.12,3942.58)PP(17.16,3951.61)PP(17.20,3962.11)PP(17.24,3974.30)PP(17.32,4007.52)PP(17.40,4048.39)PP(17.44,4068.83)PP(17.48,4087.77)PP(17.56,4125.03)PP(17.60,4142.55)PP(17.72,4189.49)PP(17.76,4204.09)PP(17.80,4218.81)PP(17.84,4234.60)PP(17.88,4250.69)PP(17.92,4266.39)PP(18.00,4298.44)PP(18.04,4311.97)PP(18.32,4395.33)PP(18.36,4408.97)PP(18.40,4423.74)PP(18.48,4447.58)PP(18.60,4476.16)PP(18.64,4484.20)PP(18.68,4491.29)PP(18.72,4498.87)PP(18.96,4554.05)PP(19.04,4574.92)PP(19.12,4599.62)PP(19.16,4612.08)PP(19.32,4659.49)PP(19.36,4670.55)PP(19.40,4682.09)PP(19.44,4693.20)PP(19.76,4784.16)PP(19.88,4814.79)PP(19.96,4834.90)PP(20.16,4890.73)PP(20.28,4922.61)PP(21.76,5240.90)PP(21.84,5260.45)PP(21.92,5279.10)PP(22.04,5314.95)PP(22.12,5341.08)PP(22.16,5354.07)PP(22.24,5376.93)PP(22.28,5388.36)

    float secant_before = (phases.y - phases.x) / (ptimes.y - ptimes.x);
    float secant_middle = (phases.z - phases.y) / (ptimes.z - ptimes.y);
    float secant_after  = (phases.w - phases.z) / (ptimes.w - ptimes.z);
    
    float tangent_before = (secant_before + secant_middle) * .5;
    float tangent_after  = (secant_middle + secant_after ) * .5;
    
    tangent_before = min(min(tangent_before, secant_before * 3.), secant_middle * 3.);
    tangent_after  = min(min(tangent_after,  secant_middle * 3.), secant_after  * 3.);
    
    float h = ptimes.z - ptimes.y;
    float t = (time - ptimes.y) / (ptimes.z - ptimes.y);
    
    float phase =
    	phases.y           * ( 2.*t*t*t - 3.*t*t + 1.) +
        h * tangent_before * (    t*t*t - 2.*t*t + t ) +
        phases.z           * (-2.*t*t*t + 3.*t*t) +
        h * tangent_after  * (    t*t*t -    t*t);
    
    float frequency =
        phases.y / h       * ( 6.*t*t - 6.*t) +
        tangent_before     * ( 3.*t*t - 4.*t + 1.) +
        phases.z / h       * (-6.*t*t + 6.*t) +
        tangent_after      * ( 3.*t*t - 2.*t);
        
    float s = 0.;

    vec4 formants_interp = mix(fprev, fnext, (time - ftimes.x) / (ftimes.y - ftimes.x));
    

    for (float harmonic = 1.; harmonic < 20.; harmonic++) {
        float falloff = exp(-.2 * harmonic);

        float harmonic_freq = harmonic * frequency;
        float resonance = 0.;
        resonance += bandpass(harmonic_freq, formants_interp.x);
        resonance += bandpass(harmonic_freq, formants_interp.y);
        resonance += bandpass(harmonic_freq, formants_interp.z);
        s += sin(TAU * phase * harmonic) * falloff * resonance * sqrt(formants_interp.w) * 5.;
    }
    //s += hash22mono(time) - hash22mono(time + 1. / iSampleRate);
    float sibilant = 0.;
    for (float sibfreq = 3500.; sibfreq < 9000.; sibfreq += 201.01) {
        sibilant += sin(TAU * (sibfreq + 1. * sin(TAU * 30. * time) / time) * time) * .03;
    }
    s += sibilant * pulse(time,  0.57, .10);
    s += sibilant * pulse(time,  1.37, .10);
    s += sibilant * pulse(time,  1.87, .10);
    s += sibilant * pulse(time,  1.97, .03);
    s += sibilant * pulse(time,  2.40, .03);
    s += sibilant * pulse(time,  3.59, .20);
    s += sibilant * pulse(time,  8.28, .08);
    s += sibilant * pulse(time,  8.66, .10);
    s += sibilant * pulse(time, 10.15, .10);
    s += sibilant * pulse(time, 13.78, .10);
    s += sibilant * pulse(time, 14.98, .03);
    s += sibilant * pulse(time, 16.46, .10);
    s += sibilant * pulse(time, 17.64, .20);
    s += sibilant * pulse(time, 18.13, .20);
    s += sibilant * pulse(time, 18.94, .03);
    s += sibilant * pulse(time, 19.54, .17);
    s += sibilant * pulse(time, 19.64, .03);
    s += sibilant * pulse(time, 21.65, .05);
    
    float glottal = sin(TAU * (1200. + 0.8 * sin(TAU * 60. * time) / time) * time);
    s += glottal * pulse(time,  7.02, .01);
    s += glottal * pulse(time,  7.37, .01);
    s += glottal * pulse(time, 10.19, .01);
    s += glottal * pulse(time, 18.57, .01);
    
    float labial = hash22mono(time) * .4;
    s += labial * pulse(time,  2.86, .03);
    s += labial * pulse(time,  5.58, .08) * 0.5;
    s += labial * pulse(time, 15.84, .03);
    s += labial * pulse(time, 20.29, .14) * 0.2;
    
    //s = sin(TAU * actualprog);*/
    //s = sin(TAU * phase);
    return vec2(s * .3/* * exp(-0.1*time) */);
}