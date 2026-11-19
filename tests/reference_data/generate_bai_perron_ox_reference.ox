/*  Reference values for the Bai-Perron procedure from the OxRegimes package.

    Runs the RgBP core of OxRegimes (github.com/knightianuncertainty/OxRegimes) on the two
    public data sets in this folder and writes every number that the Python tests compare to
    bp_oxregimes_reference.json:

      bp_simulated_consumption.csv   simulated data of the OxRegimes examples (T = 132)
      bp_exdata.csv                  example data distributed with Perron's Bai-Perron program (T = 101)

    Usage (from this folder; <packages> is the folder that contains OxRegimes):

      oxl -i<packages> generate_bai_perron_ox_reference.ox

    Break dates are written in the package convention: index (0-based) of the first observation
    of the new regime, i.e. RgBP's last-index vector plus one.  */
#include <oxstd.oxh>
#import <OxRegimes/oxregimes>

static decl s_f;

Vec(const v)
{
    decl i, s = "[";
    for (i = 0; i < sizerc(v); ++i)
        s ~= sprint(i > 0 ? "," : "", isnan(vecr(v)[i]) ? "null" : sprint("%.15g", vecr(v)[i]));
    return s ~ "]";
}

Case(const sName, const vY, const mZ, const mX, const h, const M, const dEps, const aOpt, const bLast)
{
    decl cT = rows(vY), q = columns(mZ), p = sizerc(mX) > 0 ? columns(mX) : 0, k, l;
    decl a = p > 0 ? RgBP::PartialPartitionAll(vY, mZ, mX, M, h) : RgBP::PartitionAll(RgBP::SSRTriangle(vY, mZ, h), M, h);
    decl vSSR = a[2] | a[1], vSupF = zeros(M, 1), vSeqF = constant(.NaN, M - 1, 1), vSeqAt = constant(-1, M - 1, 1);
    fprint(s_f, "\"", sName, "\":{\"T\":", cT, ",\"q\":", q, ",\"p\":", p, ",\"h\":", h, ",\"M\":", M, ",\"trimming\":", dEps,
        ",\"robust\":", aOpt[RgBP::ROBUST], ",\"prewhiten\":", aOpt[RgBP::PREWHITE], ",\"hetdat\":", aOpt[RgBP::HETDAT],
        ",\"hetvar\":", aOpt[RgBP::HETVAR], ",\"hetomega\":", aOpt[RgBP::HETOMEGA], ",\"hetq\":", aOpt[RgBP::HETQ]);
    fprint(s_f, ",\"ssr\":", Vec(vSSR), ",\"breaks_by_m\":[");
    for (k = 1; k <= M; ++k)
    {
        fprint(s_f, k > 1 ? "," : "", Vec(a[0][k - 1] + 1));
        vSupF[k - 1] = RgBP::SupF(vY, mZ, mX, a[0][k - 1], aOpt);
    }
    fprint(s_f, "],\"supf\":", Vec(vSupF));
    for (l = 1; l < M; ++l)
    {
        decl s = RgBP::SeqF(vY, mZ, mX, a[0][l - 1], h, aOpt);
        vSeqF[l - 1] = s[0];  vSeqAt[l - 1] = s[1] < 0 ? -1 : s[1] + 1;
    }
    fprint(s_f, ",\"seqf\":", Vec(vSeqF), ",\"seqf_break\":", Vec(vSeqAt));
    decl ic = RgBP::InfoCriteria(vSSR, cT, q, p);
    fprint(s_f, ",\"bic\":", Vec(ic[0]), ",\"lwz\":", Vec(ic[1]));
    decl sq = RgBP::Sequential(vY, mZ, mX, M, h, dEps, 2, aOpt, FALSE);
    fprint(s_f, ",\"seq_n\":", sq[0], ",\"seq_breaks\":", Vec(sizerc(sq[1]) > 0 ? sq[1] + 1 : <>));
    sq = RgBP::Sequential(vY, mZ, mX, M, h, dEps, 2, aOpt, TRUE);
    fprint(s_f, ",\"seq_rep_n\":", sq[0], ",\"seq_rep_breaks\":", Vec(sizerc(sq[1]) > 0 ? sq[1] + 1 : <>));
    sq = RgBP::Sequential(vY, mZ, mX, M, h, dEps, 1, aOpt, FALSE);
    fprint(s_f, ",\"seq10_n\":", sq[0], ",\"seq10_breaks\":", Vec(sizerc(sq[1]) > 0 ? sq[1] + 1 : <>));
    decl mBic = int(mincindex(ic[0]));
    fprint(s_f, ",\"bic_n\":", mBic);
    if (mBic > 0)
    {
        decl mCI = RgBP::BreakCI(vY, mZ, mX, a[0][mBic - 1], aOpt) + 1;
        fprint(s_f, ",\"ci95_lo\":", Vec(mCI[][0]), ",\"ci95_hi\":", Vec(mCI[][1]), ",\"ci90_lo\":", Vec(mCI[][2]), ",\"ci90_hi\":", Vec(mCI[][3]));
    }
    fprint(s_f, "}", bLast ? "" : ",", "\n");
}

main()
{
    decl mC = loadmat("bp_simulated_consumption.csv"), mE = loadmat("bp_exdata.csv");
    decl yc = mC[][0], zc = ones(rows(mC), 1) ~ mC[][1], one = ones(rows(mC), 1), dy = mC[][1];
    decl ye = mE[][1], ze = mE[][2], xe = mE[][0];
    decl hc = max(int(floor(0.15 * rows(mC) + 1e-9)), 3), he = 15;
    s_f = fopen("bp_oxregimes_reference.json", "w");
    fprint(s_f, "{\n");
    // options: robust, prewhite, hetdat, hetvar, hetomega, hetq
    Case("consumption_default",        yc, zc, <>, hc, 5, 0.15, {0, 0, 1, 1, 1, 1}, FALSE);
    Case("consumption_common_var",     yc, zc, <>, hc, 5, 0.15, {0, 0, 1, 0, 0, 0}, FALSE);
    Case("consumption_common_data",    yc, zc, <>, hc, 5, 0.15, {0, 0, 0, 0, 1, 1}, FALSE);
    Case("consumption_common_data_hv", yc, zc, <>, hc, 5, 0.15, {0, 0, 0, 1, 1, 0}, FALSE);
    Case("consumption_robust",         yc, zc, <>, hc, 5, 0.15, {1, 0, 1, 1, 1, 1}, FALSE);
    Case("consumption_robust_hv0",     yc, zc, <>, hc, 5, 0.15, {1, 0, 1, 0, 0, 1}, FALSE);
    Case("consumption_robust_pw",      yc, zc, <>, hc, 5, 0.15, {1, 1, 1, 1, 1, 1}, FALSE);
    Case("consumption_robust_pw_hv0",  yc, zc, <>, hc, 5, 0.15, {1, 1, 1, 0, 0, 1}, FALSE);
    Case("consumption_trim10",         yc, zc, <>, max(int(floor(0.10 * rows(mC) + 1e-9)), 3), 8, 0.10, {0, 0, 1, 1, 1, 1}, FALSE);
    Case("consumption_partial",        yc, one, dy, hc, 5, 0.15, {0, 0, 1, 1, 1, 1}, FALSE);
    Case("consumption_partial_hv0",    yc, one, dy, hc, 5, 0.15, {0, 0, 1, 0, 1, 1}, FALSE);
    Case("consumption_partial_robust", yc, one, dy, hc, 5, 0.15, {1, 1, 1, 1, 1, 1}, FALSE);
    Case("consumption_slope_partial",  yc, dy, one, hc, 5, 0.15, {0, 0, 1, 1, 1, 1}, FALSE);
    Case("exdata_pure",                ye, ze, <>, he, 5, 0.15, {0, 0, 1, 1, 1, 1}, FALSE);
    Case("exdata_pure_robust_pw",      ye, ze, <>, he, 5, 0.15, {1, 1, 1, 1, 1, 1}, FALSE);
    Case("exdata_partial",             ye, ze, xe, he, 5, 0.15, {0, 0, 1, 1, 1, 1}, TRUE);
    fprint(s_f, "}\n");
    fclose(s_f);
    println("written bp_oxregimes_reference.json");
}
