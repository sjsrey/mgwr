import numpy as np
from spglm.family import Gaussian
from spglm.glm import GLM
from .diagnostics import get_AICc


def summaryModel(self):
    summary = "=" * 75 + "\n"
    summary += "{!s:<54} {!s:>20}\n".format(
        "Model type", self.family.__class__.__name__
    )
    summary += "{!s:<60} {:14d}\n".format("Number of observations:", self.n)
    summary += "{!s:<60} {:14d}\n\n".format("Number of covariates:", self.k)
    return summary


def summaryGLM(self):

    if self.name_x is not None:
        XNames = list(self.name_x)
        if len(XNames) < self.k:
            XNames = ["Intercept"] + XNames
    else:
        XNames = ["X" + str(i) for i in range(self.k)]
    glm_rslt = GLM(self.model.y, self.model.X, constant=False, family=self.family).fit()

    summary = "{!s}\n".format("Global Regression Results")
    summary += "-" * 75 + "\n"

    if isinstance(self.family, Gaussian):
        summary += "{!s:<62} {:12.3f}\n".format(
            "Residual sum of squares:", glm_rslt.deviance
        )
        summary += "{!s:<62} {:12.3f}\n".format("Log-likelihood:", glm_rslt.llf)
        summary += "{!s:<62} {:12.3f}\n".format("AIC:", glm_rslt.aic)
        summary += "{!s:<62} {:12.3f}\n".format("AICc:", get_AICc(glm_rslt))
        summary += "{!s:<62} {:12.3f}\n".format("BIC:", glm_rslt.bic)
        summary += "{!s:<62} {:12.3f}\n".format("R2:", glm_rslt.D2)
        summary += "{!s:<62} {:12.3f}\n\n".format("Adj. R2:", glm_rslt.adj_D2)
    else:
        summary += "{!s:<62} {:12.3f}\n".format("Deviance:", glm_rslt.deviance)
        summary += "{!s:<62} {:12.3f}\n".format("Log-likelihood:", glm_rslt.llf)
        summary += "{!s:<62} {:12.3f}\n".format("AIC:", glm_rslt.aic)
        summary += "{!s:<62} {:12.3f}\n".format("AICc:", get_AICc(glm_rslt))
        summary += "{!s:<62} {:12.3f}\n".format("BIC:", glm_rslt.bic)
        summary += "{!s:<62} {:12.3f}\n".format(
            "Percent deviance explained:", glm_rslt.D2
        )
        summary += "{!s:<62} {:12.3f}\n\n".format(
            "Adj. percent deviance explained:", glm_rslt.adj_D2
        )

    summary += "{!s:<31} {!s:>10} {!s:>10} {!s:>10} {!s:>10}\n".format(
        "Variable", "Est.", "SE", "t(Est/SE)", "p-value"
    )
    summary += "{!s:<31} {!s:>10} {!s:>10} {!s:>10} {!s:>10}\n".format(
        "-" * 31, "-" * 10, "-" * 10, "-" * 10, "-" * 10
    )
    for i in range(self.k):
        summary += (
            f"{XNames[i][:30]!s:<31} {glm_rslt.params[i]:10.3f} "
            f"{glm_rslt.bse[i]:10.3f} {glm_rslt.tvalues[i]:10.3f} "
            f"{glm_rslt.pvalues[i]:10.3f}\n"
        )
    summary += "\n"
    return summary


def summaryGWR(self):
    if self.name_x is not None:
        XNames = list(self.name_x)
        if len(XNames) < self.k:
            XNames = ["Intercept"] + XNames

    else:
        XNames = ["X" + str(i) for i in range(self.k)]

    summary = "{!s}\n".format("Geographically Weighted Regression (GWR) Results")
    summary += "-" * 75 + "\n"

    if self.model.fixed:
        summary += "{!s:<50} {!s:>20}\n".format(
            "Spatial kernel:", "Fixed " + self.model.kernel
        )
    else:
        summary += "{!s:<54} {!s:>20}\n".format(
            "Spatial kernel:", "Adaptive " + self.model.kernel
        )

    summary += "{!s:<62} {:12.3f}\n".format("Bandwidth used:", self.model.bw)

    summary += "\n{!s}\n".format("Diagnostic information")
    summary += "-" * 75 + "\n"

    if isinstance(self.family, Gaussian):
        summary += "{!s:<62} {:12.3f}\n".format(
            "Residual sum of squares:", self.resid_ss
        )
        summary += "{!s:<62} {:12.3f}\n".format(
            "Effective number of parameters (trace(S)):", self.tr_S
        )
        summary += "{!s:<62} {:12.3f}\n".format(
            "Degree of freedom (n - trace(S)):", self.df_model
        )
        summary += "{!s:<62} {:12.3f}\n".format("Sigma estimate:", np.sqrt(self.sigma2))
        summary += "{!s:<62} {:12.3f}\n".format("Log-likelihood:", self.llf)
        summary += "{!s:<62} {:12.3f}\n".format("AIC:", self.aic)
        summary += "{!s:<62} {:12.3f}\n".format("AICc:", self.aicc)
        summary += "{!s:<62} {:12.3f}\n".format("BIC:", self.bic)
        summary += "{!s:<62} {:12.3f}\n".format("R2:", self.R2)
        summary += "{!s:<62} {:12.3f}\n".format("Adjusted R2:", self.adj_R2)

    else:
        summary += "{!s:<62} {:12.3f}\n".format(
            "Effective number of parameters (trace(S)):", self.tr_S
        )
        summary += "{!s:<62} {:12.3f}\n".format(
            "Degree of freedom (n - trace(S)):", self.df_model
        )
        summary += "{!s:<62} {:12.3f}\n".format("Log-likelihood:", self.llf)
        summary += "{!s:<62} {:12.3f}\n".format("AIC:", self.aic)
        summary += "{!s:<62} {:12.3f}\n".format("AICc:", self.aicc)
        summary += "{!s:<62} {:12.3f}\n".format("BIC:", self.bic)
        summary += "{!s:<60} {:12.3f}\n".format("Percent deviance explained:", self.D2)
        summary += "{!s:<60} {:12.3f}\n".format(
            "Adjusted percent deviance explained:", self.adj_D2
        )

    summary += "{!s:<62} {:12.3f}\n".format("Adj. alpha (95%):", self.adj_alpha[1])
    summary += "{!s:<62} {:12.3f}\n".format(
        "Adj. critical t value (95%):", self.critical_tval(self.adj_alpha[1])
    )

    summary += "\n{!s}\n".format("Summary Statistics For GWR Parameter Estimates")
    summary += "-" * 75 + "\n"
    summary += "{!s:<20} {!s:>10} {!s:>10} {!s:>10} {!s:>10} {!s:>10}\n".format(
        "Variable", "Mean", "STD", "Min", "Median", "Max"
    )
    summary += "{!s:<20} {!s:>10} {!s:>10} {!s:>10} {!s:>10} {!s:>10}\n".format(
        "-" * 20, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10
    )
    for i in range(self.k):
        summary += (
            f"{XNames[i][:20]!s:<20} {np.mean(self.params[:, i]):10.3f} "
            f"{np.std(self.params[:, i]):10.3f} {np.min(self.params[:, i]):10.3f} "
            f"{np.median(self.params[:, i]):10.3f} {np.max(self.params[:, i]):10.3f}\n"
        )

    summary += "=" * 75 + "\n"

    return summary


def summaryMGWR(self):

    if self.name_x is not None:
        XNames = list(self.name_x)
        if len(XNames) < self.k:
            XNames = ["Intercept"] + XNames

    else:
        XNames = ["X" + str(i) for i in range(self.k)]

    summary = ""
    summary += "{!s}\n".format(
        "Multi-Scale Geographically Weighted Regression (MGWR) Results"
    )
    summary += "-" * 75 + "\n"

    if self.model.fixed:
        summary += "{!s:<50} {!s:>20}\n".format(
            "Spatial kernel:", "Fixed " + self.model.kernel
        )
    else:
        summary += "{!s:<54} {!s:>20}\n".format(
            "Spatial kernel:", "Adaptive " + self.model.kernel
        )

    summary += "{!s:<54} {!s:>20}\n".format(
        "Criterion for optimal bandwidth:", self.model.selector.criterion
    )

    if self.model.selector.rss_score:
        summary += "{!s:<54} {!s:>20}\n".format("Score of Change (SOC) type:", "RSS")
    else:
        summary += "{!s:<54} {!s:>20}\n".format(
            "Score of Change (SOC) type:", "Smoothing f"
        )

    summary += "{!s:<54} {!s:>20}\n\n".format(
        "Termination criterion for MGWR:", self.model.selector.tol_multi
    )

    summary += "{!s}\n".format("MGWR bandwidths")
    summary += "-" * 75 + "\n"
    summary += "{!s:<15} {!s:>14} {!s:>10} {!s:>16} {!s:>16}\n".format(
        "Variable", "Bandwidth", "ENP_j", "Adj t-val(95%)", "Adj alpha(95%)"
    )
    for j in range(self.k):
        summary += (
            f"{XNames[j]!s:<14} {self.model.bws[j]:15.3f} {self.ENP_j[j]:10.3f} "
            f"{self.critical_tval()[j]:16.3f} {self.adj_alpha_j[j, 1]:16.3f}\n"
        )

    summary += "\n{!s}\n".format("Diagnostic information")
    summary += "-" * 75 + "\n"

    summary += "{!s:<62} {:12.3f}\n".format("Residual sum of squares:", self.resid_ss)
    summary += "{!s:<62} {:12.3f}\n".format(
        "Effective number of parameters (trace(S)):", self.tr_S
    )
    summary += "{!s:<62} {:12.3f}\n".format(
        "Degree of freedom (n - trace(S)):", self.df_model
    )

    summary += "{!s:<62} {:12.3f}\n".format("Sigma estimate:", np.sqrt(self.sigma2))
    summary += "{!s:<62} {:12.3f}\n".format("Log-likelihood:", self.llf)
    summary += "{!s:<62} {:12.3f}\n".format("AIC:", self.aic)
    summary += "{!s:<62} {:12.3f}\n".format("AICc:", self.aicc)
    summary += "{!s:<62} {:12.3f}\n".format("BIC:", self.bic)
    summary += "{!s:<62} {:12.3f}\n".format("R2", self.R2)
    summary += "{!s:<62} {:12.3f}\n".format("Adjusted R2", self.adj_R2)

    summary += "\n{!s}\n".format("Summary Statistics For MGWR Parameter Estimates")
    summary += "-" * 75 + "\n"
    summary += "{!s:<20} {!s:>10} {!s:>10} {!s:>10} {!s:>10} {!s:>10}\n".format(
        "Variable", "Mean", "STD", "Min", "Median", "Max"
    )
    summary += "{!s:<20} {!s:>10} {!s:>10} {!s:>10} {!s:>10} {!s:>10}\n".format(
        "-" * 20, "-" * 10, "-" * 10, "-" * 10, "-" * 10, "-" * 10
    )
    for i in range(self.k):
        summary += (
            f"{XNames[i][:20]!s:<20} {np.mean(self.params[:, i]):10.3f} "
            f"{np.std(self.params[:, i]):10.3f} {np.min(self.params[:, i]):10.3f} "
            f"{np.median(self.params[:, i]):10.3f} {np.max(self.params[:, i]):10.3f}\n"
        )

    summary += "=" * 75 + "\n"
    return summary
