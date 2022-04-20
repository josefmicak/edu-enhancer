using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authentication.Google;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;

namespace ViewLayer.Controllers
{
    [Authorize]
    public class AccountController : Controller
    {
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
                return BadRequest(); // TODO: Handle this better.
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
                        StudentController studentController = new StudentController();
                        List<(string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email)> students = studentController.LoadStudentsByEmail();
                        if(students.Count > 0)
                        {
                            try
                            {
                                (string loginEmail, string studentNumberIdentifier, int role, string studentIdentifier, string login, string firstName, string lastName, string email) student = studentController.LoadStudentByEmail(claimsIdentity.Claims.ToList()[2].Value);

                                switch (student.role)
                                {
                                    case 2:
                                        return RedirectToAction("ManageUserList", "Home");
                                    case 1:
                                        return RedirectToAction("TeacherMenu", "Home");
                                    default:
                                        return RedirectToAction("BrowseSolvedTestList", "Home", new { studentIdentifier = student.studentIdentifier });
                                }
                            }
                            catch
                            {
                                return RedirectToAction("Index", "Home", new { error = "user_not_found_exception" });
                            }
                        }
                        else
                        {
                            studentController.EditUser(claimsIdentity.Claims.ToList()[2].Value, "", 2);
                            return RedirectToAction("ManageUserList", "Home");
                        }
                    }
                }
            }
            return RedirectToAction("Index", "Home", new { error = "unexpected_exception" });
        }

        /// <summary>
        /// Google Login Sign out
        /// </summary>
        /// <returns></returns>
        [AllowAnonymous]
        public async Task<IActionResult> SignOut()
        {
            await HttpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
            
            return RedirectToAction("Index", "Home");
        }
    }
}
