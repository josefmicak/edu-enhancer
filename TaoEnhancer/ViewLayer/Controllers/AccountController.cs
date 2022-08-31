using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authentication.Google;
using Microsoft.AspNetCore.Authorization;
using System.Security.Claims;
using DomainModel;
using DataLayer;
using System.Security.Principal;

namespace ViewLayer.Controllers
{
    [Authorize]
    public class AccountController : Controller
    {
        private readonly CourseContext _context;

        public AccountController(CourseContext context)
        {
            _context = context;
        }

        /// <summary>
        /// Google Login Redirection To Google Login Page
        /// </summary>
        /// <returns></returns>
        [AllowAnonymous]
        public IActionResult SignIn()
        {
            return new ChallengeResult(
                GoogleDefaults.AuthenticationScheme,
                new AuthenticationProperties
                {
                    RedirectUri = Url.Action("GoogleResponse", "Account") // Where google responds back
                });
        }

        /// <summary>
        /// Google Login Response After Login Operation From Google Page
        /// </summary>
        /// <returns></returns>
        [AllowAnonymous]
        public async Task<IActionResult> GoogleResponse()
        {
            //Check authentication response as mentioned on startup file as o.DefaultSignInScheme = "External"
            var authenticateResult = await HttpContext.AuthenticateAsync("Google");
            if (!authenticateResult.Succeeded)
                return BadRequest();
            //Check if the redirection has been done via google or any other links
            if (authenticateResult.Principal.Identities.ToList()[0].AuthenticationType.ToLower() == "google")
            {
                //check if principal value exists or not 
                if (authenticateResult.Principal != null)
                {
                    //get google account id for any operation to be carried out on the basis of the id
                    var googleAccountId = authenticateResult.Principal.FindFirst(ClaimTypes.NameIdentifier)?.Value;
                    //claim value initialization as mentioned on the startup file with o.DefaultScheme = "Application"
                    var claimsIdentity = new ClaimsIdentity(CookieAuthenticationDefaults.AuthenticationScheme);
                    if (authenticateResult.Principal != null)
                    {
                        //Now add the values on claim and redirect to the page to be accessed after successful login
                        var details = authenticateResult.Principal.Claims.ToList();
                        claimsIdentity.AddClaim(authenticateResult.Principal.FindFirst(ClaimTypes.NameIdentifier)); // Unique ID Of The User
                        claimsIdentity.AddClaim(authenticateResult.Principal.FindFirst(ClaimTypes.Name)); // Full Name Of The User
                        claimsIdentity.AddClaim(authenticateResult.Principal.FindFirst(ClaimTypes.Email)); // Email Address of The User
                        await HttpContext.SignInAsync(CookieAuthenticationDefaults.AuthenticationScheme, new ClaimsPrincipal(claimsIdentity));

                        // Redirect after login
                        return AfterSignInRedirect(claimsIdentity);
                    }
                }
            }
            return RedirectToAction("Index", "Home", new { error = "unexpected_exception" });
        }

        public IActionResult AfterSignInRedirect(ClaimsIdentity claimsIdentity)
        {
            User user = _context.Users.FirstOrDefault(u => u.Email == claimsIdentity.Claims.ToList()[2].Value);
            Student student = _context.Students.FirstOrDefault(s => s.Email == claimsIdentity.Claims.ToList()[2].Value);
            if (user == null && student == null)
            {
                string fullName = claimsIdentity.Claims.ToList()[1].Value;
                string[] fullNameSplitBySpace = fullName.Split(" ");
                Common.Config.Application["firstName"] = fullNameSplitBySpace[0];
                Common.Config.Application["lastName"] = fullNameSplitBySpace[1];
                Common.Config.Application["email"] = claimsIdentity.Claims.ToList()[2].Value;
                return RedirectToAction("UserRegistration", "Home");
            }
            else
            {
                if(user != null)
                {
                    Common.Config.Application["login"] = user.Login;
                    switch (user.Role)
                    {
                        case 2:
                            return RedirectToAction("TeacherMenu", "Home");
                        case 3:
                            return RedirectToAction("AdminMenu", "Home");
                        case 4:
                            return RedirectToAction("MainAdminMenu", "Home");
                        default:
                            break;
                    }
                }
                else if(student != null)
                {
                    Common.Config.Application["login"] = student.Login;
                    return RedirectToAction("BrowseSolvedTestList", "Home");
                }
                //todo: throw exception - no user found
                return RedirectToAction("Index", "Home");
            }
        }

        /// <summary>
        /// Used when testing mode is on - allows the user to browse pages that require authorization
        /// </summary>
        [AllowAnonymous]
        public async Task<IActionResult> TestingModeLogin(string name, string email)
        {
            var claimsIdentity = new ClaimsIdentity(CookieAuthenticationDefaults.AuthenticationScheme);
            claimsIdentity.AddClaim(new Claim(ClaimTypes.NameIdentifier, email));
            claimsIdentity.AddClaim(new Claim(ClaimTypes.Name, name));
            claimsIdentity.AddClaim(new Claim(ClaimTypes.Email, email));
            await HttpContext.SignInAsync(CookieAuthenticationDefaults.AuthenticationScheme, new ClaimsPrincipal(claimsIdentity));

            return AfterSignInRedirect(claimsIdentity);
        }

        /// <summary>
        /// Google Login Sign out
        /// </summary>
        public async Task<IActionResult> GoogleSignOut()
        {
            await HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);

            return RedirectToAction("Index", "Home");
        }
    }
}
